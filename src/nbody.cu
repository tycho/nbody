/*
 *
 * nbody.cu
 *
 * N-body example that illustrates gravitational simulation.
 * This is the type of computation that GPUs excel at:
 * parallelizable, with lots of FLOPS per unit of external
 * memory bandwidth required.
 *
 * Copyright (c) 2011-2012, Archaea Software, LLC.
 * Copyright (c) 2012-2021, Uplink Laboratories, LLC.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <getopt.h>

#ifdef _WIN32
#include <conio.h>
#endif
#ifdef USE_GL
#include <SDL2/SDL_main.h>
#endif

#include <cmath>

#include "chThread.h"
#include "chError.h"

#include "nbody.h"
#include "nbody_render_gl.h"
#include "nbody_util.h"

#include "nbody_CPU_AOS.h"
#include "nbody_CPU_AOS_tiled.h"
#include "nbody_CPU_SOA.h"
#include "nbody_CPU_SOA_tiled.h"
#include "nbody_CPU_SIMD.h"

#ifdef USE_CUDA
#include "nbody_GPU_AOS.h"
#include "nbody_GPU_AOS_const.h"
//#include "nbody_GPU_AOS_tiled.h"
//#include "nbody_GPU_AOS_tiled_const.h"
//#include "nbody_GPU_SOA_tiled.h"
#include "nbody_GPU_Shared.h"
#include "nbody_GPU_Shuffle.h"
//#include "nbody_GPU_Atomic.h"
#endif

using namespace std;

#define SOA_INTEGRATION 1

#define DEFAULT_KPARTICLES 16
#define DEFAULT_SEED 7

static const algorithm_def_t s_algorithms[] = {
    { "CPU_SOA",             ALGORITHM_SOA,           { .soa = ComputeGravitation_SOA                 } },
    { "CPU_SOA_tiled",       ALGORITHM_SOA,           { .soa = ComputeGravitation_SOA_tiled           } },
#ifdef HAVE_SIMD
    { SIMD_ALGORITHM_NAME,   ALGORITHM_SOA,           { .soa = ComputeGravitation_SIMD                } },
#endif
    { "CPU_AOS",             ALGORITHM_AOS,           { .aos = ComputeGravitation_AOS                 } },
    { "CPU_AOS_tiled",       ALGORITHM_AOS,           { .aos = ComputeGravitation_AOS_tiled           } },
#ifdef USE_CUDA
    { "GPU_AOS",             ALGORITHM_AOS_GPU,       { .aos = ComputeGravitation_GPU_AOS             } },
    { "GPU_Shared",          ALGORITHM_AOS_GPU,       { .aos = ComputeGravitation_GPU_Shared          } },
    { "GPU_Const",           ALGORITHM_AOS_GPU,       { .aos = ComputeGravitation_GPU_AOS_const       } },
    { "MultiGPU",            ALGORITHM_AOS_MULTIGPU,  { .aos = ComputeGravitation_multiGPU            } },
    { "GPU_Shuffle",         ALGORITHM_AOS_GPU,       { .aos = ComputeGravitation_GPU_Shuffle         } },
//    { "GPU_SOA_tiled",       ALGORITHM_AOS_GPU,       { .aos = ComputeGravitation_GPU_SOA_tiled       } },
//    { "GPU_AOS_tiled",       ALGORITHM_AOS_GPU,       { .aos = ComputeGravitation_GPU_AOS_tiled       } },
//    { "GPU_AOS_tiled_const", ALGORITHM_AOS_GPU,       { .aos = ComputeGravitation_GPU_AOS_tiled_const } },
//    { "GPU_Atomic",          ALGORITHM_AOS_GPU,       { .aos = ComputeGravitation_GPU_Atomic          } },
#endif
    { 0 },
};

static inline bool isGPUAlgorithm(const algorithm_def_t *algorithm)
{
    return (algorithm->type == ALGORITHM_AOS_GPU || algorithm->type == ALGORITHM_AOS_MULTIGPU);
}

static int maxAlgorithmIdx(void)
{
    static int idx = -1;
    if (idx != -1)
        return idx;
    for (idx = 0; s_algorithms[idx].name; idx++);
    idx--;
    return idx;
}

static float
relError( float a, float b )
{
    if ( a == b ) return 0.0f;
    return fabsf(a-b)/b;
}

static int g_bCUDAPresent;

float *g_hostAOS_PosMass;
float *g_hostAOS_VelInvMass;
float *g_hostAOS_Force;

#ifdef USE_CUDA
static float *g_dptrAOS_PosMass;
static float *g_dptrAOS_Force;
#endif

// Buffer to hold the golden version of the forces, used for comparison
// Along with timing results, we report the maximum relative error with
// respect to this array.
static float *g_hostAOS_Force_Golden;

float *g_hostSOA_Pos[3];
float *g_hostSOA_Force[3];
float *g_hostSOA_Vel[3];
float *g_hostSOA_Mass;
float *g_hostSOA_InvMass;

size_t g_N;

static float g_softening = 0.1f;
static float g_damping = 0.95f;
static float g_timestep = 0.016f;
//static float g_scale = 1.54f;
static float g_scale = 0.9f;
static float g_velocityScale = 8.0f;

static inline void
integrateOne(
    float * __restrict pos,
    float * __restrict vel,
    float * const __restrict force,
    float invMass,
    float dt,
    float damping
)
{
    // acceleration = force / mass;
    // new velocity = old velocity + acceleration * deltaTime
    vel[0] += (force[0] * invMass) * dt;
    vel[1] += (force[1] * invMass) * dt;
    vel[2] += (force[2] * invMass) * dt;

    vel[0] *= damping;
    vel[1] *= damping;
    vel[2] *= damping;

    // new position = old position + velocity * deltaTime
    pos[0] += vel[0] * dt;
    pos[1] += vel[1] * dt;
    pos[2] += vel[2] * dt;
}

#if SOA_INTEGRATION

static NOINLINE void
integrateGravitation_SOA(
    float ** __restrict ppos,
    float *  __restrict pinvmass,
    float ** __restrict pvel,
    float ** const __restrict pforce,
    float dt,
    float damping,
    size_t N
)
{
    ASSERT_ALIGNED(pinvmass, NBODY_ALIGNMENT);
    ASSERT_ALIGNED(ppos[0], NBODY_ALIGNMENT);
    ASSERT_ALIGNED(ppos[1], NBODY_ALIGNMENT);
    ASSERT_ALIGNED(ppos[2], NBODY_ALIGNMENT);
    ASSERT_ALIGNED(pforce[0], NBODY_ALIGNMENT);
    ASSERT_ALIGNED(pforce[1], NBODY_ALIGNMENT);
    ASSERT_ALIGNED(pforce[2], NBODY_ALIGNMENT);

    ASSUME(N >= 1024);
    ASSUME(N % 1024 == 0);

#pragma omp simd
    for ( size_t i = 0; i < N; i++ ) {
        float pos[3], vel[3], force[3];

        pos[0] = ppos[0][i];
        pos[1] = ppos[1][i];
        pos[2] = ppos[2][i];

        float invMass = pinvmass[i];

        vel[0] = pvel[0][i];
        vel[1] = pvel[1][i];
        vel[2] = pvel[2][i];

        force[0] = pforce[0][i];
        force[1] = pforce[1][i];
        force[2] = pforce[2][i];

        integrateOne(pos, vel, force, invMass, dt, damping);

        ppos[0][i] = pos[0];
        ppos[1][i] = pos[1];
        ppos[2][i] = pos[2];

        pvel[0][i] = vel[0];
        pvel[1][i] = vel[1];
        pvel[2][i] = vel[2];
    }
}

#else

static NOINLINE void
integrateGravitation_AOS(
    float * __restrict ppos,
    float * __restrict pvel,
    float * const __restrict pforce,
    float dt,
    float damping,
    size_t N
)
{
    ASSERT_ALIGNED(ppos, NBODY_ALIGNMENT);
    ASSERT_ALIGNED(pvel, NBODY_ALIGNMENT);
    ASSERT_ALIGNED(pforce, NBODY_ALIGNMENT);

    ASSUME(N >= 1024);
    ASSUME(N % 1024 == 0);

#pragma omp simd
    for ( size_t i = 0; i < N; i++ ) {
        const int index = 4*i;

        float pos[3], vel[3], force[3];
        pos[0] = ppos[index+0];
        pos[1] = ppos[index+1];
        pos[2] = ppos[index+2];

        float invMass = pvel[index+3];

        vel[0] = pvel[index+0];
        vel[1] = pvel[index+1];
        vel[2] = pvel[index+2];

        force[0] = pforce[index+0];
        force[1] = pforce[index+1];
        force[2] = pforce[index+2];

        integrateOne(pos, vel, force, invMass, dt, damping);

        ppos[index+0] = pos[0];
        ppos[index+1] = pos[1];
        ppos[index+2] = pos[2];

        pvel[index+0] = vel[0];
        pvel[index+1] = vel[1];
        pvel[index+2] = vel[2];
    }
}

#endif

static int g_bCrossCheck = 1;
static int g_bNoCPU = 0;

static NOINLINE int
ComputeGravitation(
    float *ms,
    float *maxRelError,
    const algorithm_def_t *algorithm,
    int bCrossCheck )
{
#ifdef USE_CUDA
    cudaError_t status;
#endif
    int bSOA = 0;

    bool bIsGPUAlgorithm = isGPUAlgorithm(algorithm);

    if (g_bNoCPU && !bIsGPUAlgorithm)
        return 1;

    if (!g_bCUDAPresent && bIsGPUAlgorithm)
        return 1;

    if ( bCrossCheck && algorithm != &s_algorithms[0] ) {
        ComputeGravitation_SOA(
                        g_hostSOA_Force,
                        g_hostSOA_Pos,
                        g_hostSOA_Mass,
                        g_softening*g_softening,
                        g_N );
        for ( size_t i = 0; i < g_N; i++ ) {
            g_hostAOS_Force_Golden[4*i+0] = g_hostSOA_Force[0][i];
            g_hostAOS_Force_Golden[4*i+1] = g_hostSOA_Force[1][i];
            g_hostAOS_Force_Golden[4*i+2] = g_hostSOA_Force[2][i];
        }
    }

    /* Reset the force values so we know the function tested did work. */
    memset(g_hostAOS_Force,    0, g_N * sizeof(float) * 4);
    memset(g_hostSOA_Force[0], 0, g_N * sizeof(float));
    memset(g_hostSOA_Force[1], 0, g_N * sizeof(float));
    memset(g_hostSOA_Force[2], 0, g_N * sizeof(float));

    switch ( algorithm->type ) {
        case ALGORITHM_SOA:
            *ms = algorithm->soa(
                g_hostSOA_Force,
                g_hostSOA_Pos,
                g_hostSOA_Mass,
                g_softening*g_softening,
                g_N );
            bSOA = 1;
            break;
        case ALGORITHM_AOS:
            *ms = algorithm->aos(
                g_hostAOS_Force,
                g_hostAOS_PosMass,
                g_softening*g_softening,
                g_N );
            break;
#ifdef USE_CUDA
        case ALGORITHM_AOS_GPU:
            CUDART_CHECK( cudaMemcpyAsync(
                g_dptrAOS_PosMass,
                g_hostAOS_PosMass,
                4*g_N*sizeof(float),
                cudaMemcpyHostToDevice ) );
            CUDART_CHECK( cudaMemset(
                g_dptrAOS_Force,
                0,
                4*g_N*sizeof(float) ) );
            *ms = algorithm->aos(
                g_dptrAOS_Force,
                g_dptrAOS_PosMass,
                g_softening*g_softening,
                g_N );
            CUDART_CHECK( cudaMemcpy(
                g_hostAOS_Force,
                g_dptrAOS_Force,
                4*g_N*sizeof(float),
                cudaMemcpyDeviceToHost ) );
            break;
        case ALGORITHM_AOS_MULTIGPU:
            *ms = algorithm->aos(
                g_hostAOS_Force,
                g_hostAOS_PosMass,
                g_softening*g_softening,
                g_N );
            break;
#endif
        default:
            fprintf(stderr, "Unrecognized algorithm index: %d\n", algorithm->type);
            abort();
    }

    if ( *ms < __FLT_EPSILON__ )
        return 1;

    if ( bSOA ) {
#pragma omp simd
        for ( size_t i = 0; i < g_N; i++ ) {
            g_hostAOS_Force[4*i+0] = g_hostSOA_Force[0][i];
            g_hostAOS_Force[4*i+1] = g_hostSOA_Force[1][i];
            g_hostAOS_Force[4*i+2] = g_hostSOA_Force[2][i];
        }
    } else {
#if SOA_INTEGRATION
#pragma omp simd
        for ( size_t i = 0; i < g_N; i++ ) {
            g_hostSOA_Force[0][i] = g_hostAOS_Force[4*i+0];
            g_hostSOA_Force[1][i] = g_hostAOS_Force[4*i+1];
            g_hostSOA_Force[2][i] = g_hostAOS_Force[4*i+2];
        }
#endif
    }

    *maxRelError = 0.0f;
    if ( bCrossCheck && algorithm != &s_algorithms[0] ) {
        float max = 0.0f;
        for ( size_t i = 0; i < 4*g_N; i++ ) {
            if ((i + 1) % 4 == 0)
                continue;
            float err = relError( g_hostAOS_Force[i], g_hostAOS_Force_Golden[i] );
            if ( err > max ) {
                max = err;
            }
        }
        *maxRelError = max;
    }

#if SOA_INTEGRATION
    integrateGravitation_SOA(
        g_hostSOA_Pos,
        g_hostSOA_InvMass,
        g_hostSOA_Vel,
        g_hostSOA_Force,
        g_timestep,
        g_damping,
        g_N );

    // SOA -> AOS data structures in case we are measuring AOS performance
#pragma omp simd
    for ( size_t i = 0; i < g_N; i++ ) {
        g_hostAOS_PosMass[4*i+0]    = g_hostSOA_Pos[0][i];
        g_hostAOS_PosMass[4*i+1]    = g_hostSOA_Pos[1][i];
        g_hostAOS_PosMass[4*i+2]    = g_hostSOA_Pos[2][i];
        g_hostAOS_VelInvMass[4*i+0] = g_hostSOA_Vel[0][i];
        g_hostAOS_VelInvMass[4*i+1] = g_hostSOA_Vel[1][i];
        g_hostAOS_VelInvMass[4*i+2] = g_hostSOA_Vel[2][i];
        g_hostAOS_VelInvMass[4*i+3] = g_hostSOA_InvMass[i];
        g_hostAOS_PosMass[4*i+3]    = g_hostSOA_Mass[i];
    }
#else
    integrateGravitation_AOS(
        g_hostAOS_PosMass,
        g_hostAOS_VelInvMass,
        g_hostAOS_Force,
        g_timestep,
        g_damping,
        g_N );

    // AOS -> SOA data structures in case we are measuring SOA performance
#pragma omp simd
    for ( size_t i = 0; i < g_N; i++ ) {
        g_hostSOA_Pos[0][i]  = g_hostAOS_PosMass[4*i+0];
        g_hostSOA_Pos[1][i]  = g_hostAOS_PosMass[4*i+1];
        g_hostSOA_Pos[2][i]  = g_hostAOS_PosMass[4*i+2];
        g_hostSOA_Vel[0][i]  = g_hostAOS_VelInvMass[4*i+0];
        g_hostSOA_Vel[1][i]  = g_hostAOS_VelInvMass[4*i+1];
        g_hostSOA_Vel[2][i]  = g_hostAOS_VelInvMass[4*i+2];
        g_hostSOA_InvMass[i] = g_hostAOS_VelInvMass[4*i+3];
        g_hostSOA_Mass[i]    = g_hostAOS_PosMass[4*i+3];
    }
#endif

    return 0;
#ifdef USE_CUDA
Error:
    return 1;
#endif
}

static worker_thread *g_GPUThreadPool;
static worker_thread g_renderThread;
static int g_bRunning = 1;
int g_maxGPUs;
int g_numGPUs;

struct gpuInit_struct
{
    int iGPU;

    cudaError_t status;
};

static int initializeGPU( void *_p )
{
    cudaError_t status;

    struct gpuInit_struct *p = (struct gpuInit_struct *) _p;
    CUDART_CHECK( cudaSetDevice( p->iGPU ) );
    CUDART_CHECK( cudaFree(0) );
Error:
    p->status = status;
    return 0;
}

static int teardownGPU( void *_p )
{
    cudaError_t status;

    struct gpuInit_struct *p = (struct gpuInit_struct *) _p;
    CUDART_CHECK( cudaSetDevice( p->iGPU ) );
    CUDART_CHECK( cudaDeviceReset() );
Error:
    p->status = status;
    return 0;
}

static int freeArrays(void)
{
#ifdef USE_CUDA
    cudaError_t status;

    if ( g_bCUDAPresent ) {
        CUDART_CHECK( cudaDeviceSynchronize() );
        CUDART_CHECK( cudaFreeHost( g_hostAOS_PosMass ) );
        for ( size_t i = 0; i < 3; i++ ) {
            CUDART_CHECK( cudaFreeHost( g_hostSOA_Pos[i] ) );
            CUDART_CHECK( cudaFreeHost( g_hostSOA_Force[i] ) );
            CUDART_CHECK( cudaFreeHost( g_hostSOA_Vel[i] ) );
        }
        CUDART_CHECK( cudaFreeHost( g_hostAOS_Force ) );
        CUDART_CHECK( cudaFreeHost( g_hostAOS_Force_Golden ) );
        CUDART_CHECK( cudaFreeHost( g_hostAOS_VelInvMass ) );
        CUDART_CHECK( cudaFreeHost( g_hostSOA_Mass ) );
        CUDART_CHECK( cudaFreeHost( g_hostSOA_InvMass ) );

        CUDART_CHECK( cudaFree( g_dptrAOS_PosMass ) );
        CUDART_CHECK( cudaFree( g_dptrAOS_Force ) );
    } else
#endif
    {
        alignedFree(g_hostAOS_PosMass);
        for ( size_t i = 0; i < 3; i++ ) {
            alignedFree(g_hostSOA_Pos[i]);
            alignedFree(g_hostSOA_Force[i]);
            alignedFree(g_hostSOA_Vel[i]);
        }
        alignedFree(g_hostAOS_Force);
        alignedFree(g_hostAOS_Force_Golden);
        alignedFree(g_hostAOS_VelInvMass);
        alignedFree(g_hostSOA_Mass);
        alignedFree(g_hostSOA_InvMass);
    }
    return 0;
#ifdef USE_CUDA
Error:
    fprintf(stderr, "Failed to clean up memory.\n");
    return 1;
#endif
}

static int allocArrays(void)
{
#ifdef USE_CUDA
    cudaError_t status;

    if ( g_bCUDAPresent ) {
        CUDART_CHECK( cudaHostAlloc( (void **) &g_hostAOS_PosMass, 4*g_N*sizeof(float), cudaHostAllocPortable|cudaHostAllocMapped ) );
        for ( size_t i = 0; i < 3; i++ ) {
            CUDART_CHECK( cudaHostAlloc( (void **) &g_hostSOA_Pos[i], g_N*sizeof(float), cudaHostAllocPortable|cudaHostAllocMapped ) );
            CUDART_CHECK( cudaHostAlloc( (void **) &g_hostSOA_Force[i], g_N*sizeof(float), cudaHostAllocPortable|cudaHostAllocMapped ) );
            CUDART_CHECK( cudaHostAlloc( (void **) &g_hostSOA_Vel[i], g_N*sizeof(float), cudaHostAllocPortable|cudaHostAllocMapped ) );
        }
        CUDART_CHECK( cudaHostAlloc( (void **) &g_hostAOS_Force, 4*g_N*sizeof(float), cudaHostAllocPortable|cudaHostAllocMapped ) );
        CUDART_CHECK( cudaHostAlloc( (void **) &g_hostAOS_Force_Golden, 4*g_N*sizeof(float), cudaHostAllocPortable|cudaHostAllocMapped ) );
        CUDART_CHECK( cudaHostAlloc( (void **) &g_hostAOS_VelInvMass, 4*g_N*sizeof(float), cudaHostAllocPortable|cudaHostAllocMapped ) );
        CUDART_CHECK( cudaHostAlloc( (void **) &g_hostSOA_Mass, g_N*sizeof(float), cudaHostAllocPortable|cudaHostAllocMapped ) );
        CUDART_CHECK( cudaHostAlloc( (void **) &g_hostSOA_InvMass, g_N*sizeof(float), cudaHostAllocPortable|cudaHostAllocMapped ) );

        CUDART_CHECK( cudaMalloc( &g_dptrAOS_PosMass, 4*g_N*sizeof(float) ) );
        CUDART_CHECK( cudaMalloc( (void **) &g_dptrAOS_Force, 4*g_N*sizeof(float) ) );
    } else
#endif
    {
        g_hostAOS_PosMass = (float *)alignedAlloc(NBODY_ALIGNMENT, sizeof(float) * 4 * g_N);
        if (!g_hostAOS_PosMass)
            goto Error;

        for ( size_t i = 0; i < 3; i++ ) {
            g_hostSOA_Pos[i] = (float *)alignedAlloc(NBODY_ALIGNMENT, sizeof(float) * g_N);
            if (!g_hostSOA_Pos[i])
                goto Error;

            g_hostSOA_Force[i] = (float *)alignedAlloc(NBODY_ALIGNMENT, sizeof(float) * g_N);
            if (!g_hostSOA_Force[i])
                goto Error;

            g_hostSOA_Vel[i] = (float *)alignedAlloc(NBODY_ALIGNMENT, sizeof(float) * g_N);
            if (!g_hostSOA_Vel[i])
                goto Error;
        }
        g_hostSOA_Mass = (float *)alignedAlloc(NBODY_ALIGNMENT, sizeof(float) * g_N);
        if (!g_hostSOA_Mass)
            goto Error;

        g_hostAOS_Force = (float *)alignedAlloc(NBODY_ALIGNMENT, sizeof(float) * 4 * g_N);
        if (!g_hostAOS_Force)
            goto Error;

        g_hostAOS_Force_Golden = (float *)alignedAlloc(NBODY_ALIGNMENT, sizeof(float) * 4 * g_N);
        if (!g_hostAOS_Force_Golden)
            goto Error;

        g_hostAOS_VelInvMass = (float *)alignedAlloc(NBODY_ALIGNMENT, sizeof(float) * 4 * g_N);
        if (!g_hostAOS_VelInvMass)
            goto Error;

        g_hostSOA_InvMass = (float *)alignedAlloc(NBODY_ALIGNMENT, sizeof(float) * g_N);
        if (!g_hostSOA_InvMass)
            goto Error;
    }
    return 0;
Error:
    fprintf(stderr, "Failed to allocate required memory.\n");
    return 1;
}

static void print_algorithms(void)
{
    int idx = 0;
    bool bGPUsAvailable = g_maxGPUs > 0;
    bool bGPUsEnabled = g_numGPUs > 0;
    fprintf(stderr, "\nAlgorithms available in this build:\n\n");
    for (idx = 0; s_algorithms[idx].name; idx++) {
        const char *suffix= "";
        bool bIsGPUAlgorithm = isGPUAlgorithm(&s_algorithms[idx]);
        if (bIsGPUAlgorithm) {
            if (!bGPUsAvailable)
                suffix = " [disabled, no GPUs available]";
            else if (!bGPUsEnabled)
                suffix = " [disabled by user]";
        } else {
            if (g_bNoCPU)
                suffix = " [disabled by user]";
        }
        fprintf(stdout, "   %d - %s%s\n", idx, s_algorithms[idx].name, suffix);
    }
#ifndef USE_CUDA
    fprintf(stderr, "\nThis build does not have CUDA support enabled. All GPU algorithms are unavailable.\n");
#endif
    fprintf(stderr, "\n");
}

static void print_usage(const char *argv0)
{
    fprintf(stderr, "Usage: nbody [arguments]\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Available arguments:\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "    --bodies=<N> | -n <N>\n");
    fprintf(stderr, "        Specifies the number of random bodies to use in the simulation. The\n");
    fprintf(stderr, "        number is multiplied by 1024. [default: %d]\n", DEFAULT_KPARTICLES);
    fprintf(stderr, "\n");
    fprintf(stderr, "    --gpus=<N> | -g <N>\n");
    fprintf(stderr, "        Specifies the number of GPUs to use for the GPU-based algorithms.\n");
    fprintf(stderr, "        [default: number of available GPUs]\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "    --no-cpu\n");
    fprintf(stderr, "        Disables all CPU-based simulations (including crosscheck). Only makes\n");
    fprintf(stderr, "        sense if GPU-based algorithms are available.\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "    --no-crosscheck\n");
    fprintf(stderr, "        Disables cross-validation of results against a CPU implementation.\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "    --iterations=<N> | -i <N>\n");
    fprintf(stderr, "        Specifies the number of iterations through the algorithm list.\n");
    fprintf(stderr, "        [default: loop forever]\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "    --cycle-after=<N> | -c <N>\n");
    fprintf(stderr, "        Specifies the number of simulations steps to execute before cycling to\n");
    fprintf(stderr, "        the next available algorithm. [default: none, don't cycle]\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "    --seed=<N> | -s <N>\n");
    fprintf(stderr, "        Specifies the seed value for the random number generatior. This value\n");
    fprintf(stderr, "        determines the initial positions for the bodies in the system. [default: %d]\n", DEFAULT_SEED);
    fprintf(stderr, "\n");
#ifdef USE_GL
    /* We only print the --graphics help section if we've got OpenGL support
     * compiled in. But we always accept the command line argument for
     * compatibility with other builds.
     */
    fprintf(stderr, "    --graphics\n");
    fprintf(stderr, "        Enables OpenGL rendering, if available.\n");
    fprintf(stderr, "\n");
#endif
    fprintf(stderr, "    --list\n");
    fprintf(stderr, "        Lists the available simulation algorithms.\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "    --algorithm=<name or index> | -a <name or index>\n");
    fprintf(stderr, "        Specifies a specific algorithm name or index to start with. See --list\n");
    fprintf(stderr, "        for the list of available algorithms for this argument.\n");
    fprintf(stderr, "        [default: %s]\n", s_algorithms[0].name);
    fprintf(stderr, "\n");
    fprintf(stderr, "    --verbose\n");
    fprintf(stderr, "        By default, the output is rate limited to ten lines per seocnd. If this\n");
    fprintf(stderr, "        argument is specified, a status update is printed for every iteration.\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "    --help\n");
    fprintf(stderr, "        Prints this help text.\n");
}

static semaphore g_semRenderInitDone;

static int render_loop(void *_unused)
{
    auto start = chrono::steady_clock::now();
    MAYBE_UNUSED uint32_t frames = 0;
    gl_init_window();
    g_semRenderInitDone.post();
    while(g_bRunning) {
        if (gl_display() != 0)
            break;
        frames++;
        auto now = chrono::steady_clock::now();
        if (chrono::duration_cast<chrono::seconds>(now - start).count() > 2)
        {
#ifdef _DEBUG
            fprintf(stderr, "%0.1f FPS\n", frames / 2.0);
#endif
            frames = 0;
            start = now;
        }
    }
    gl_quit();
    return 0;
}

static void render_init(void)
{
    g_renderThread.delegate(render_loop, (void*)1, false);
    g_semRenderInitDone.wait();
}

int main(int argc, char **argv)
{
    cudaError_t status;

    // kiloparticles
    int kParticles = DEFAULT_KPARTICLES, maxIterations = 0, cycleAfter = 0;
    int name_width = 0;
    int idxFirstAlgorithm = 0;
    int bPrintListOnly = 0;
    int bUseGraphics = 0;
    int bVerbose = 0;
    int numThreads;
    unsigned int nSeed = DEFAULT_SEED;

    static const struct option cli_options[] = {
        { "bodies", required_argument, NULL, 'n' },
        { "gpus", required_argument, NULL, 'g' },
        { "no-cpu", no_argument, &g_bNoCPU, 1 },
        { "no-crosscheck", no_argument, &g_bCrossCheck, 0 },
        { "iterations", required_argument, NULL, 'i' },
        { "cycle-after", required_argument, NULL, 'c' },
        { "seed", required_argument, NULL, 's' },
        { "list", no_argument, NULL, 'l' },
        { "algorithm", required_argument, NULL, 'a' },
        { "graphics", no_argument, NULL, 'G' },
        { "verbose", no_argument, NULL, 'v' },
        { "help", no_argument, NULL, 'h' },
        { NULL, 0, NULL, 0 }
    };

    status = cudaGetDeviceCount( &g_maxGPUs );
    if (status != cudaSuccess)
        g_numGPUs = 0;
    else
        g_numGPUs = g_maxGPUs;

    while (1) {
        int option = getopt_long(argc, argv, "n:s:i:c:g:la:Gvh", cli_options, NULL);

        if (option == -1)
            break;

        switch (option) {
        case 'c':
            {
                int v;
                if (sscanf(optarg, "%d", &v) != 1) {
                    fprintf(stderr, "ERROR: Couldn't parse integer argument for '--cycle-after'\n");
                    return 1;
                }
                if (v < 1) {
                    fprintf(stderr, "ERROR: Requested cycle size less than 1\n");
                    return 1;
                }
                cycleAfter = v;
            }
            break;
        case 'i':
            {
                int v;
                if (sscanf(optarg, "%d", &v) != 1) {
                    fprintf(stderr, "ERROR: Couldn't parse integer argument for '--iterations'\n");
                    return 1;
                }
                if (v < 1) {
                    fprintf(stderr, "ERROR: Requested number of iterations less than 1\n");
                    return 1;
                }
                maxIterations = v;
            }
            break;
        case 'n':
            {
                int v;
                if (sscanf(optarg, "%d", &v) != 1) {
                    fprintf(stderr, "ERROR: Couldn't parse integer argument for '--bodies'\n");
                    return 1;
                }
                if (v < 1) {
                    fprintf(stderr, "ERROR: Requested number of bodies less than 1\n");
                    return 1;
                }
                kParticles = v;
            }
            break;
        case 's':
            {
                unsigned int v;
                if (sscanf(optarg, "%u", &v) != 1) {
                    fprintf(stderr, "ERROR: Couldn't parse integer argument for '--seed'\n");
                    return 1;
                }
                if (v < 1) {
                    fprintf(stderr, "ERROR: Seed must be nonzero integer value\n");
                    return 1;
                }
                nSeed = v;
            }
            break;
        case 'g':
            {
                int v;
                if (sscanf(optarg, "%d", &v) != 1) {
                    fprintf(stderr, "ERROR: Couldn't parse integer argument for '--gpus'\n");
                    return 1;
                }
                if (g_maxGPUs < 1) {
                    if (v == 0) {
                        g_numGPUs = 0;
                        break;
                    }
                    fprintf(stderr, "ERROR: No GPUs available, cannot handle '--gpus' argument.\n");
                    return 1;
                }
                if (v < 1) {
                    fprintf(stderr, "Requested number of GPUs less than 1, disabling GPU algorithms.\n");
                    v = 0;
                }
                if (v > g_maxGPUs) {
                    fprintf(stderr, "Requested %d GPUs, but only have %d, using all available GPUs.\n",
                            v, g_maxGPUs);
                    v = g_maxGPUs;
                }
                g_numGPUs = v;
            }
            break;
        case 'a':
            {
                const char *type;
                int v;

                idxFirstAlgorithm = -1;

                // First try to parse as an integer argument
                if (sscanf(optarg, "%d", &v) == 1) {
                    type = "index";
                    if (v >= 0 && v <= maxAlgorithmIdx()) {
                        idxFirstAlgorithm = v;
                    }
                } else {
                    type = "name";
                    // Alternatively, check if it's a valid algorithm name
                    for (v = 0; s_algorithms[v].name; v++) {
                        int n;
                        n = strcasecmp(optarg, s_algorithms[v].name);
                        if (n == 0) {
                            // Exact match
                            idxFirstAlgorithm = v;
                            break;
                        }
                        n = strncasecmp(optarg, s_algorithms[v].name, strlen(optarg));
                        if (n == 0) {
                            // Partial match, keep searching in case there's an
                            // exact match
                            idxFirstAlgorithm = v;
                            continue;
                        }
                    }
                }

                if (idxFirstAlgorithm == -1) {
                    fprintf(stderr, "Invalid algorithm %s '%s'\n", type, optarg);
                    print_algorithms();
                    return 1;
                }
            }
            break;
        case 'G':
            bUseGraphics = 1;
            break;
        case 'l':
            bPrintListOnly = 1;
            break;
        case 'v':
            bVerbose = 1;
            break;
        case 'h':
        case '?':
            print_usage(argv[0]);
            return 1;
        }
    }

    // for reproducible results for a given N
    seedRandom(nSeed);

    g_bCUDAPresent = g_numGPUs > 0;
    if ( g_bCUDAPresent ) {
        struct cudaDeviceProp prop;
        CUDART_CHECK( cudaGetDeviceProperties( &prop, 0 ) );
    }

    if ( g_bNoCPU ) {
        if ( !g_bCUDAPresent ) {
            fprintf(stderr, "ERROR: --no-cpu specified, but CUDA disabled or not available\n" );
            goto cleanup;
        }
    }

#ifndef USE_GL
    if ( bUseGraphics ) {
        /* This is a soft error. Things still work without graphics. */
        fprintf(stderr, "Graphics requested, but OpenGL support not compiled in.\n");
        bUseGraphics = 0;
    }
#endif

    if (bPrintListOnly) {
        print_algorithms();
        return 0;
    }

    if ( g_numGPUs ) {
        g_GPUThreadPool = new worker_thread[g_numGPUs];
        for ( int i = 0; i < g_numGPUs; i++ ) {
            struct gpuInit_struct initGPU = {i};
            g_GPUThreadPool[i].delegate(initializeGPU, &initGPU, true);
            if ( cudaSuccess != initGPU.status ) {
                fprintf(stderr, "Initializing GPU %d failed "
                    " with %d (%s)\n",
                    i,
                    initGPU.status,
                    cudaGetErrorString( initGPU.status ));
                return 1;
            }
        }
    }

    if ( g_bNoCPU ) {
        g_bCrossCheck = 0;
    }

    g_N = kParticles * 1024;

    numThreads = processorCount();

    fprintf(stderr, "Running simulation with %u particles", (unsigned int)g_N);

    if (g_bCrossCheck)
        fprintf(stderr, ", crosscheck enabled");

    if (g_bNoCPU)
        fprintf(stderr, ", CPU disabled");
    else
        fprintf(stderr, ", %d CPU threads", numThreads);

    if (g_numGPUs)
        fprintf(stderr, ", up to %u GPUs", g_numGPUs);

    fprintf(stderr, "\n");

    if (allocArrays() != 0)
        return 1;

    for ( const algorithm_def_t *algorithm = s_algorithms; algorithm->name; algorithm++ )
    {
        int this_name_len = strlen(algorithm->name);
        if (this_name_len > name_width)
            name_width = this_name_len;
    }

    randomUnitBodies( g_hostAOS_PosMass, g_hostAOS_VelInvMass, g_N, g_scale, g_velocityScale );
    for ( size_t i = 0; i < g_N; i++ ) {
        g_hostSOA_Pos[0][i] = g_hostAOS_PosMass[4*i+0];
        g_hostSOA_Pos[1][i] = g_hostAOS_PosMass[4*i+1];
        g_hostSOA_Pos[2][i] = g_hostAOS_PosMass[4*i+2];
        g_hostSOA_Mass[i] = g_hostAOS_PosMass[4*i+3];
        g_hostSOA_Vel[0][i] = g_hostAOS_VelInvMass[4*i+0];
        g_hostSOA_Vel[1][i] = g_hostAOS_VelInvMass[4*i+1];
        g_hostSOA_Vel[2][i] = g_hostAOS_VelInvMass[4*i+2];
        g_hostSOA_InvMass[i] = g_hostAOS_VelInvMass[4*i+3];
    }

    {
        int algorithm_idx = idxFirstAlgorithm;
        int steps = 0, iterations = 0;
        int bStop = 0;
        auto start = chrono::steady_clock::now();
        if ( bUseGraphics )
            render_init();
        while ( !bStop ) {
            char ch;
            float ms, err;
            const algorithm_def_t *algorithm = &s_algorithms[algorithm_idx];
            auto now = chrono::steady_clock::now();

            if (ComputeGravitation(&ms, &err, algorithm, g_bCrossCheck))
                goto next_algorithm;

            if (bVerbose || chrono::duration_cast<chrono::milliseconds>(now - start).count() >= 100)
            {
                double interactionsPerSecond = (double) g_N*g_N*1000.0f / ms,
                       flops = interactionsPerSecond * (3 + 6 + 4 + 1 + 6) * 1e-3;

                start = now;

                if ( interactionsPerSecond > 1e9 )
                {
                    fprintf(stdout, "\r%*s: %8.2f ms = %8.3fx10^9 interactions/s (%9.2lf GFLOPS)",
                        name_width,
                        algorithm->name,
                        ms,
                        interactionsPerSecond/1e9,
                        flops * 1e-6 );
                }
                else
                {
                    fprintf(stdout, "\r%*s: %8.2f ms = %8.3fx10^6 interactions/s (%9.2lf GFLOPS)",
                        name_width,
                        algorithm->name,
                        ms,
                        interactionsPerSecond/1e6,
                        flops * 1e-6 );
                }
                if (g_bCrossCheck)
                    fprintf(stdout, " (Rel. error: %E)\n", err );
                else
                    fprintf(stdout, "\n" );
            }

            steps++;
            if (cycleAfter && steps % cycleAfter == 0) {
next_algorithm:
                steps = 0;
                algorithm_idx++;
                if ( !s_algorithms[algorithm_idx].name ) {
                    algorithm_idx = 0;
                    iterations++;
                }
            } else if (!cycleAfter) {
                iterations++;
            }
            if (maxIterations && iterations >= maxIterations) {
                bStop = 1;
            }
            ch = gl_getch();
            if ( !ch && kbhit() )
                ch = getch();
            switch ( ch ) {
                case ' ':
                    algorithm_idx++;
                    if ( !s_algorithms[algorithm_idx].name ) {
                        algorithm_idx = 0;
                        iterations++;
                    }
                    break;
                case 'q':
                case 'Q':
                    bStop = 1;
                    break;
                default:
                    break;
            }
        }
    }

cleanup:
    g_bRunning = 0;
    g_renderThread.stop();

    freeArrays();

    for ( int i = 0; i < g_numGPUs; i++ ) {
        struct gpuInit_struct initGPU = {i};
        g_GPUThreadPool[i].delegate(teardownGPU, &initGPU, true);
        if ( cudaSuccess != initGPU.status ) {
            fprintf(stderr, "GPU %d teardown failed "
                " with %d (%s)\n",
                i,
                initGPU.status,
                cudaGetErrorString( initGPU.status ) );
            return 1;
        }
    }

    for (int i = 0; i < g_numGPUs; i++) {
        g_GPUThreadPool[i].stop();
    }

    delete [] g_GPUThreadPool;

    return 0;
Error:
    if ( cudaSuccess != status ) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString( status ) );
    }
    return 1;
}

/* vim: set ts=4 sts=4 sw=4 et: */
