/*
 *
 * nbody.cu
 *
 * N-body example that illustrates gravitational simulation.
 * This is the type of computation that GPUs excel at:
 * parallelizable, with lots of FLOPS per unit of external
 * memory bandwidth required.
 *
 * Requires: No minimum SM requirement.  If SM 3.x is not available,
 * this application quietly replaces the shuffle and fast-atomic
 * implementations with the shared memory implementation.
 *
 * Copyright (c) 2011-2012, Archaea Software, LLC.
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

#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _WIN32
#include <conio.h>
#define valloc malloc

#pragma comment (lib, "libtime.lib")
#pragma comment (lib, "libc11.lib")
#else

#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/select.h>

static int kbhit(void)
{
    struct termios oldt, newt;
    int ch;
    int oldf;

    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

    ch = getchar();

    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    fcntl(STDIN_FILENO, F_SETFL, oldf);

    if(ch != EOF)
    {
        ungetc(ch, stdin);
        return 1;
    }

    return 0;
}

// we only call getch() when kbhit() has told us there
// is a pending keystroke
static int
getch(void)
{
    return getchar();
}

#endif

#include <math.h>

#include "libtime.h"

#include "chThread.h"
#include "chError.h"

#include "nbody.h"
#include "nbody_util.h"

enum nbodyAlgorithm_enum {
	CPU_AOS = 0,    /* This is the golden implementation */
	CPU_AOS_tiled,
	CPU_SOA,
	CPU_SOA_tiled,
#ifdef HAVE_SIMD
	CPU_SIMD,
#endif
	GPU_AOS,
	GPU_Shared,
	GPU_Const,
	multiGPU,
	// SM 3.0 only
	GPU_Shuffle,
	//GPU_AOS_tiled,
	//GPU_AOS_tiled_const,
	//    GPU_Atomic
};

static const char *rgszAlgorithmNames[] = {
	"CPU_AOS",
	"CPU_AOS_tiled",
	"CPU_SOA",
	"CPU_SOA_tiled",
#ifdef HAVE_SIMD
#if defined(__ALTIVEC__)
	"AltiVec intrin",
#elif defined(__ARM_NEON__)
	"NEON intrin",
#elif defined(__AVX__)
	"AVX intrin",
#elif defined(__SSE__)
	"SSE intrin",
#else
#error "Define a name for this platform's SIMD"
#endif
#endif
	"GPU_AOS",
	"GPU_Shared",
	"GPU_Const",
	"multiGPU",
	// SM 3.0 only
	"GPU_Shuffle",
	//"GPU_AOS_tiled",
	//"GPU_AOS_tiled_const",
	//    "GPU_Atomic"
};

static unsigned int holdrand;

static inline void seedRandom( unsigned int seed )
{
    srandom(seed);
}

static inline int nbodyRandom( void )
{
    uint32_t v;
#if defined(HIGH_ENTROPY) && defined __RDRND__
    int i = _rdrand32_step(&v);
    if (!i)
        abort();
#else
    v = random();
#endif
    return v;
}

static inline void
randomVector( float v[3] )
{
    float lenSqr;
    do {
        v[0] = nbodyRandom() / (float) RAND_MAX * 2 - 1;
        v[1] = nbodyRandom() / (float) RAND_MAX * 2 - 1;
        v[2] = nbodyRandom() / (float) RAND_MAX * 2 - 1;
        lenSqr = v[0]*v[0]+v[1]*v[1]+v[2]*v[2];
    } while ( lenSqr > 1.0f );
}

static void
randomUnitBodies( float *pos, float *vel, size_t N )
{
    for ( size_t i = 0; i < N; i++ ) {
        randomVector( &pos[4*i] );
        randomVector( &vel[4*i] );
        pos[4*i+3] = 1.0f;  // unit mass
        vel[4*i+3] = 1.0f;
    }
}

static float
relError( float a, float b )
{
    if ( a == b ) return 0.0f;
    return fabsf(a-b)/b;
}

static int g_bCUDAPresent;
static int g_bSM30Present;

float *g_hostAOS_PosMass;
float *g_hostAOS_VelInvMass;
float *g_hostAOS_Force;

#ifndef NO_CUDA
static float *g_dptrAOS_PosMass;
static float *g_dptrAOS_Force;
#endif

// Buffer to hold the golden version of the forces, used for comparison
// Along with timing results, we report the maximum relative error with
// respect to this array.
static float *g_hostAOS_Force_Golden;

float *g_hostSOA_Pos[3];
float *g_hostSOA_Force[3];
float *g_hostSOA_Mass;
float *g_hostSOA_InvMass;

static size_t g_N;

static float g_softening = 0.1f;
static float g_damping = 0.995f;
static float g_dt = 0.016f;

#include "nbody_CPU_AOS.h"
#include "nbody_CPU_AOS_tiled.h"
#include "nbody_CPU_SOA.h"
#include "nbody_CPU_SOA_tiled.h"
#include "nbody_CPU_SIMD.h"

#ifndef NO_CUDA
#include "bodybodyInteraction.cuh"
#include "nbody_GPU_AOS.cuh"
#include "nbody_GPU_AOS_const.cuh"
//#include "nbody_GPU_AOS_tiled.cuh"
//#include "nbody_GPU_AOS_tiled_const.cuh"
//#include "nbody_GPU_SOA_tiled.cuh"
#include "nbody_GPU_Shuffle.cuh"
#include "nbody_GPU_Atomic.cuh"
#endif

static void
integrateGravitation_AOS( float *ppos, float *pvel, float *pforce, float dt, float damping, size_t N )
{
    for ( size_t i = 0; i < N; i++ ) {
        const int index = 4*i;
        const int indexForce = 3*i;

        float pos[3], vel[3], force[3];
        pos[0] = ppos[index+0];
        pos[1] = ppos[index+1];
        pos[2] = ppos[index+2];
        float invMass = pvel[index+3];

        vel[0] = pvel[index+0];
        vel[1] = pvel[index+1];
        vel[2] = pvel[index+2];

        force[0] = pforce[indexForce+0];
        force[1] = pforce[indexForce+1];
        force[2] = pforce[indexForce+2];

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

        ppos[index+0] = pos[0];
        ppos[index+1] = pos[1];
        ppos[index+2] = pos[2];

        pvel[index+0] = vel[0];
        pvel[index+1] = vel[1];
        pvel[index+2] = vel[2];
    }
}

static enum nbodyAlgorithm_enum g_Algorithm;

//
// g_maxAlgorithm is used to determine when to rotate g_Algorithm back to CPU_AOS
// If CUDA is present, it depends on SM version
//
// The shuffle and tiled implementations are SM 3.0 only.
//
// The CPU and GPU algorithms must be contiguous, and the logic in main() to
// initialize this value must be modified if any new algorithms are added.
//
static enum nbodyAlgorithm_enum g_maxAlgorithm;
static int g_bCrossCheck = 1;
static int g_bNoCPU = 0;

static int
ComputeGravitation(
    float *ms,
    float *maxRelError,
    enum nbodyAlgorithm_enum algorithm,
    int bCrossCheck )
{
#ifndef NO_CUDA
    cudaError_t status;
#endif
    int bSOA = 0;

    // AOS -> SOA data structures in case we are measuring SOA performance
    for ( size_t i = 0; i < g_N; i++ ) {
        g_hostSOA_Pos[0][i]  = g_hostAOS_PosMass[4*i+0];
        g_hostSOA_Pos[1][i]  = g_hostAOS_PosMass[4*i+1];
        g_hostSOA_Pos[2][i]  = g_hostAOS_PosMass[4*i+2];
        g_hostSOA_Mass[i]    = g_hostAOS_PosMass[4*i+3];
        g_hostSOA_InvMass[i] = 1.0f / g_hostSOA_Mass[i];
    }

    if ( bCrossCheck ) {
        ComputeGravitation_SOA(
                        g_hostSOA_Force,
                        g_hostSOA_Pos,
                        g_hostSOA_Mass,
                        g_softening*g_softening,
                        g_N );
        for ( size_t i = 0; i < g_N; i++ ) {
            g_hostAOS_Force_Golden[3*i+0] = g_hostSOA_Force[0][i];
            g_hostAOS_Force_Golden[3*i+1] = g_hostSOA_Force[1][i];
            g_hostAOS_Force_Golden[3*i+2] = g_hostSOA_Force[2][i];
        }
    }

    /* Reset the force values so we know the function tested did work. */
    memset(g_hostAOS_Force,    0, g_N * sizeof(float) * 3);
    memset(g_hostSOA_Force[0], 0, g_N * sizeof(float));
    memset(g_hostSOA_Force[1], 0, g_N * sizeof(float));
    memset(g_hostSOA_Force[2], 0, g_N * sizeof(float));

#ifndef NO_CUDA
    // CPU->GPU copies in case we are measuring GPU performance
    if ( g_bCUDAPresent ) {
        CUDART_CHECK( cudaMemcpyAsync(
            g_dptrAOS_PosMass,
            g_hostAOS_PosMass,
            4*g_N*sizeof(float),
            cudaMemcpyHostToDevice ) );
    }
#endif

    switch ( algorithm ) {
        case CPU_AOS:
            *ms = ComputeGravitation_AOS(
                g_hostAOS_Force,
                g_hostAOS_PosMass,
                g_softening*g_softening,
                g_N );
            break;
        case CPU_AOS_tiled:
            *ms = ComputeGravitation_AOS_tiled(
                g_hostAOS_Force,
                g_hostAOS_PosMass,
                g_softening*g_softening,
                g_N );
            break;
        case CPU_SOA:
            *ms = ComputeGravitation_SOA(
                g_hostSOA_Force,
                g_hostSOA_Pos,
                g_hostSOA_Mass,
                g_softening*g_softening,
                g_N );
            bSOA = 1;
            break;
        case CPU_SOA_tiled:
            *ms = ComputeGravitation_SOA_tiled(
                g_hostSOA_Force,
                g_hostSOA_Pos,
                g_hostSOA_Mass,
                g_softening*g_softening,
                g_N );
            bSOA = 1;
            break;
#ifdef HAVE_SIMD
        case CPU_SIMD:
            *ms = ComputeGravitation_SIMD(
                g_hostSOA_Force,
                g_hostSOA_Pos,
                g_hostSOA_Mass,
                g_softening*g_softening,
                g_N );
            bSOA = 1;
            break;
#endif
#ifndef NO_CUDA
        case GPU_AOS:
            *ms = ComputeGravitation_GPU_AOS(
                g_dptrAOS_Force,
                g_dptrAOS_PosMass,
                g_softening*g_softening,
                g_N );
            CUDART_CHECK( cudaMemcpy( g_hostAOS_Force, g_dptrAOS_Force, 3*g_N*sizeof(float), cudaMemcpyDeviceToHost ) );
            break;
            /*
        case GPU_AOS_tiled:
            *ms = ComputeGravitation_GPU_AOS_tiled(
                g_dptrAOS_Force,
                g_dptrAOS_PosMass,
                g_softening*g_softening,
                g_N );
            CUDART_CHECK( cudaMemcpy( g_hostAOS_Force, g_dptrAOS_Force, 3*g_N*sizeof(float), cudaMemcpyDeviceToHost ) );
            break;
        case GPU_AOS_tiled_const:
            *ms = ComputeGravitation_GPU_AOS_tiled_const(
                g_dptrAOS_Force,
                g_dptrAOS_PosMass,
                g_softening*g_softening,
                g_N );
            CUDART_CHECK( cudaMemcpy( g_hostAOS_Force, g_dptrAOS_Force, 3*g_N*sizeof(float), cudaMemcpyDeviceToHost ) );
            break;
            */
#if 0
// commented out - too slow even on SM 3.0
        case GPU_Atomic:
            CUDART_CHECK( cudaMemset( g_dptrAOS_Force, 0, 3*sizeof(float) ) );
            *ms = ComputeGravitation_GPU_Atomic(
                g_dptrAOS_Force,
                g_dptrAOS_PosMass,
                g_softening*g_softening,
                g_N );
            CUDART_CHECK( cudaMemcpy( g_hostAOS_Force, g_dptrAOS_Force, 3*g_N*sizeof(float), cudaMemcpyDeviceToHost ) );
            break;
#endif
        case GPU_Shared:
            CUDART_CHECK( cudaMemset( g_dptrAOS_Force, 0, 3*g_N*sizeof(float) ) );
            *ms = ComputeGravitation_GPU_Shared(
                g_dptrAOS_Force,
                g_dptrAOS_PosMass,
                g_softening*g_softening,
                g_N );
            CUDART_CHECK( cudaMemcpy( g_hostAOS_Force, g_dptrAOS_Force, 3*g_N*sizeof(float), cudaMemcpyDeviceToHost ) );
            break;
        case GPU_Const:
            CUDART_CHECK( cudaMemset( g_dptrAOS_Force, 0, 3*g_N*sizeof(float) ) );
            *ms = ComputeNBodyGravitation_GPU_AOS_const(
                g_dptrAOS_Force,
                g_dptrAOS_PosMass,
                g_softening*g_softening,
                g_N );
            CUDART_CHECK( cudaMemcpy( g_hostAOS_Force, g_dptrAOS_Force, 3*g_N*sizeof(float), cudaMemcpyDeviceToHost ) );
            break;
        case GPU_Shuffle:
            CUDART_CHECK( cudaMemset( g_dptrAOS_Force, 0, 3*g_N*sizeof(float) ) );
            *ms = ComputeGravitation_GPU_Shuffle(
                g_dptrAOS_Force,
                g_dptrAOS_PosMass,
                g_softening*g_softening,
                g_N );
            CUDART_CHECK( cudaMemcpy( g_hostAOS_Force, g_dptrAOS_Force, 3*g_N*sizeof(float), cudaMemcpyDeviceToHost ) );
            break;
        case multiGPU:
            memset( g_hostAOS_Force, 0, 3*g_N*sizeof(float) );
            *ms = ComputeGravitation_multiGPU(
                g_hostAOS_Force,
                g_hostAOS_PosMass,
                g_softening*g_softening,
                g_N );
            break;
#endif
        default:
            fprintf(stderr, "Unrecognized algorithm index: %d\n", algorithm);
            abort();
    }

    // SOA -> AOS
    if ( bSOA ) {
        for ( size_t i = 0; i < g_N; i++ ) {
            g_hostAOS_Force[3*i+0] = g_hostSOA_Force[0][i];
            g_hostAOS_Force[3*i+1] = g_hostSOA_Force[1][i];
            g_hostAOS_Force[3*i+2] = g_hostSOA_Force[2][i];
        }
    }

    *maxRelError = 0.0f;
    if ( bCrossCheck ) {
        float max = 0.0f;
        for ( size_t i = 0; i < 3*g_N; i++ ) {
            float err = relError( g_hostAOS_Force[i], g_hostAOS_Force_Golden[i] );
            if ( err > max ) {
                max = err;
            }
        }
        *maxRelError = max;
    }

    integrateGravitation_AOS(
        g_hostAOS_PosMass,
        g_hostAOS_VelInvMass,
        g_hostAOS_Force,
        g_dt,
        g_damping,
        g_N );
    return 0;
#ifndef NO_CUDA
Error:
    return 1;
#endif
}

static worker_thread_t *g_GPUThreadPool;
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

static int
freeArrays(void)
{
#ifndef NO_CUDA
    cudaError_t status;

    if ( g_bCUDAPresent ) {
        CUDART_CHECK( cudaDeviceSynchronize() );
        CUDART_CHECK( cudaFreeHost( g_hostAOS_PosMass ) );
        for ( size_t i = 0; i < 3; i++ ) {
            CUDART_CHECK( cudaFreeHost( g_hostSOA_Pos[i] ) );
            CUDART_CHECK( cudaFreeHost( g_hostSOA_Force[i] ) );
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
        free(g_hostAOS_PosMass);
        for ( size_t i = 0; i < 3; i++ ) {
            free(g_hostSOA_Pos[i]);
            free(g_hostSOA_Force[i]);
        }
        free(g_hostAOS_Force);
        free(g_hostAOS_Force_Golden);
        free(g_hostAOS_VelInvMass);
        free(g_hostSOA_Mass);
        free(g_hostSOA_InvMass);
    }
    return 0;
#ifndef NO_CUDA
Error:
    fprintf(stderr, "Failed to clean up memory.\n");
    return 1;
#endif
}

static int
allocArrays(void)
{
#ifndef NO_CUDA
    cudaError_t status;

    if ( g_bCUDAPresent ) {
        CUDART_CHECK( cudaHostAlloc( (void **) &g_hostAOS_PosMass, 4*g_N*sizeof(float), cudaHostAllocPortable|cudaHostAllocMapped ) );
        for ( size_t i = 0; i < 3; i++ ) {
            CUDART_CHECK( cudaHostAlloc( (void **) &g_hostSOA_Pos[i], g_N*sizeof(float), cudaHostAllocPortable|cudaHostAllocMapped ) );
            CUDART_CHECK( cudaHostAlloc( (void **) &g_hostSOA_Force[i], g_N*sizeof(float), cudaHostAllocPortable|cudaHostAllocMapped ) );
        }
        CUDART_CHECK( cudaHostAlloc( (void **) &g_hostAOS_Force, 3*g_N*sizeof(float), cudaHostAllocPortable|cudaHostAllocMapped ) );
        CUDART_CHECK( cudaHostAlloc( (void **) &g_hostAOS_Force_Golden, 3*g_N*sizeof(float), cudaHostAllocPortable|cudaHostAllocMapped ) );
        CUDART_CHECK( cudaHostAlloc( (void **) &g_hostAOS_VelInvMass, 4*g_N*sizeof(float), cudaHostAllocPortable|cudaHostAllocMapped ) );
        CUDART_CHECK( cudaHostAlloc( (void **) &g_hostSOA_Mass, g_N*sizeof(float), cudaHostAllocPortable|cudaHostAllocMapped ) );
        CUDART_CHECK( cudaHostAlloc( (void **) &g_hostSOA_InvMass, g_N*sizeof(float), cudaHostAllocPortable|cudaHostAllocMapped ) );

        CUDART_CHECK( cudaMalloc( &g_dptrAOS_PosMass, 4*g_N*sizeof(float) ) );
        CUDART_CHECK( cudaMalloc( (void **) &g_dptrAOS_Force, 3*g_N*sizeof(float) ) );
    } else
#endif
    {
        g_hostAOS_PosMass = (float *)valloc(sizeof(float) * 4 * g_N);
        if (!g_hostAOS_PosMass)
            goto Error;

        for ( size_t i = 0; i < 3; i++ ) {
            g_hostSOA_Pos[i] = (float *)valloc(sizeof(float) * g_N);
            if (!g_hostSOA_Pos[i])
                goto Error;

            g_hostSOA_Force[i] = (float *)valloc(sizeof(float) * g_N);
            if (!g_hostSOA_Force[i])
                goto Error;
        }
        g_hostSOA_Mass = (float *)valloc(sizeof(float) * g_N);
        if (!g_hostSOA_Mass)
            goto Error;

        g_hostAOS_Force = (float *)valloc(sizeof(float) * 3 * g_N);
        if (!g_hostAOS_Force)
            goto Error;

        g_hostAOS_Force_Golden = (float *)valloc(sizeof(float) * 3 * g_N);
        if (!g_hostAOS_Force_Golden)
            goto Error;

        g_hostAOS_VelInvMass = (float *)valloc(sizeof(float) * 4 * g_N);
        if (!g_hostAOS_VelInvMass)
            goto Error;

        g_hostSOA_InvMass = (float *)valloc(sizeof(float) * g_N);
        if (!g_hostSOA_InvMass)
            goto Error;
    }
    return 0;
Error:
    fprintf(stderr, "Failed to allocate required memory.\n");
    return 1;
}

static void usage(const char *argv0)
{
    printf( "Usage: nbody --bodies=N [--gpus=N] [--no-cpu] [--no-crosscheck] [--cycle-after=N] [--iterations=N]\n" );
    printf( "    --bodies is multiplied by 1024 (default is 16)\n" );
    printf( "    By default, the app checks results against a CPU implementation; \n" );
    printf( "    disable this behavior with --no-crosscheck.\n" );
    printf( "    The CPU implementation may be disabled with --no-cpu.\n" );
    printf( "    --no-cpu implies --no-crosscheck.\n\n" );
    printf( "    --iterations specifies a fixed number of iterations to execute\n" );
    printf( "    --cycle-after specifies the number of iterations before rotating\n" );
    printf( "                  to the next available algorithm\n" );
}

int
main( int argc, char *argv[] )
{
    cudaError_t status;
    // kiloparticles
    int kParticles = 16, maxIterations = 0, cycleAfter = 0;
    static enum nbodyAlgorithm_enum firstAlgorithm;

    static const struct option cli_options[] = {
        { "bodies", required_argument, NULL, 'b' },
        { "gpus", required_argument, NULL, 'g' },
        { "no-cpu", no_argument, &g_bNoCPU, 1 },
        { "no-crosscheck", no_argument, &g_bCrossCheck, 0 },
        { "iterations", required_argument, NULL, 'i' },
        { "cycle-after", required_argument, NULL, 'c' },
        { "help", no_argument, NULL, 'h' },
        { NULL, 0, NULL, 0 }
    };

    libtime_init();

    status = cudaGetDeviceCount( &g_numGPUs );
    if (status != cudaSuccess)
        g_numGPUs = 0;

    while (1) {
        int option = getopt_long(argc, argv, "n:i:c:", cli_options, NULL);

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
        case 'b':
            {
                int v;
                if (sscanf(optarg, "%d", &v) != 1) {
                    fprintf(stderr, "ERROR: Couldn't parse integer argument for '--bodies'\n");
                    return 1;
                }
                if (v < 1) {
                    printf("ERROR: Requested number of bodies less than 1");
                    return 1;
                }
                kParticles = v;
            }
            break;
        case 'g':
            {
                int v;
                if (sscanf(optarg, "%d", &v) != 1) {
                    fprintf(stderr, "ERROR: Couldn't parse integer argument for '--gpus'\n");
                    return 1;
                }
                if (v < 1) {
                    if (g_numGPUs > 0)
                        fprintf(stderr, "Requested number of GPUs less than 1, disabling GPU algorithms.\n");
                    g_numGPUs = 0;
                    break;
                }
                if (v > g_numGPUs) {
                    fprintf(stderr, "Requested %d GPUs, but only have %d, using all available GPUs.\n",
                            v, g_numGPUs);
                    break;
                }
                g_numGPUs = v;
            }
            break;
        case 'h':
        case '?':
            usage(argv[0]);
            return 1;
        }
    }

    // for reproducible results for a given N
    seedRandom(7);

    g_bCUDAPresent = g_numGPUs > 0;
    if ( g_bCUDAPresent ) {
        struct cudaDeviceProp prop;
        CUDART_CHECK( cudaGetDeviceProperties( &prop, 0 ) );
        g_bSM30Present = prop.major >= 3;
    }

    if ( g_bNoCPU && ! g_bCUDAPresent ) {
        fprintf(stderr, "ERROR: --no-cpu specified, but no CUDA present\n" );
        exit(1);
    }

    if ( g_numGPUs ) {
        g_GPUThreadPool = (worker_thread_t *)malloc(sizeof(worker_thread_t) * g_numGPUs);
        for (int i = 0; i < g_numGPUs; i++) {
            if (worker_create(&g_GPUThreadPool[i])) {
                fprintf( stderr, "Error initializing thread pool\n" );
                return 1;
            }
            if (worker_start(&g_GPUThreadPool[i])) {
                fprintf( stderr, "Error starting thread pool\n" );
                return 1;
            }
        }
        for ( int i = 0; i < g_numGPUs; i++ ) {
            struct gpuInit_struct initGPU = {i};
            worker_delegate(&g_GPUThreadPool[i], initializeGPU, &initGPU, 1);
            if ( cudaSuccess != initGPU.status ) {
                fprintf( stderr, "Initializing GPU %d failed "
                    " with %d (%s)\n",
                    i,
                    initGPU.status,
                    cudaGetErrorString( initGPU.status ) );
                return 1;
            }
        }
    }

    if ( g_bNoCPU ) {
        g_bCrossCheck = 0;
    }

    g_N = kParticles * 1024;

    printf( "Running simulation with %d particles, crosscheck %s, CPU %s, %d threads\n", (int) g_N,
        g_bCrossCheck ? "enabled" : "disabled",
        g_bNoCPU ? "disabled" : "enabled",
        processorCount() );

#if defined(HAVE_SIMD)
    g_maxAlgorithm = CPU_SIMD;
#else
    g_maxAlgorithm = CPU_SOA_tiled;
#endif
    g_Algorithm = firstAlgorithm = g_bCUDAPresent ? GPU_AOS : CPU_SOA;
    if ( g_bCUDAPresent || g_bNoCPU ) {
        // max algorithm is different depending on whether SM 3.0 is present
        g_maxAlgorithm = g_bSM30Present ? GPU_Shuffle : multiGPU;
    }

    if (allocArrays() != 0)
        return 1;

    randomUnitBodies( g_hostAOS_PosMass, g_hostAOS_VelInvMass, g_N );
    for ( size_t i = 0; i < g_N; i++ ) {
        g_hostSOA_Mass[i] = g_hostAOS_PosMass[4*i+3];
        g_hostSOA_InvMass[i] = 1.0f / g_hostSOA_Mass[i];
    }

    {
        int steps = 0, iterations = 0;
        int bStop = 0;
        while ( ! bStop ) {
            float ms, err;

            if (ComputeGravitation(&ms, &err, g_Algorithm, g_bCrossCheck)) {
                fprintf( stderr, "Error computing timestep\n" );
                exit(1);
            }
            double interactionsPerSecond = (double) g_N*g_N*1000.0f / ms,
                   flops = interactionsPerSecond * (3 + 6 + 4 + 1 + 6) * 1e-3;
            if ( interactionsPerSecond > 1e9 ) {
                printf ( "\r%13s: %8.2f ms = %8.3fx10^9 interactions/s (%9.2lf GFLOPS)",
                    rgszAlgorithmNames[g_Algorithm],
                    ms,
                    interactionsPerSecond/1e9,
                    flops * 1e-6 );
            }
            else {
                printf ( "\r%13s: %8.2f ms = %8.3fx10^6 interactions/s (%9.2lf GFLOPS)",
                    rgszAlgorithmNames[g_Algorithm],
                    ms,
                    interactionsPerSecond/1e6,
                    flops * 1e-6 );
            }
            if (g_bCrossCheck)
                printf( " (Rel. error: %E)\n", err );
            else
                printf( "\n" );

            steps++;
            if (cycleAfter && steps % cycleAfter == 0) {
                g_Algorithm = (enum nbodyAlgorithm_enum) (g_Algorithm+1);
                if ( g_Algorithm > g_maxAlgorithm ) {
                    g_Algorithm = g_bNoCPU ? GPU_AOS : CPU_AOS;
                }
                if ( g_Algorithm == firstAlgorithm) {
                    iterations++;
                }
            } else if (!cycleAfter) {
                iterations++;
            }
            if (maxIterations && iterations >= maxIterations) {
                bStop = 1;
            }
            if ( kbhit() ) {
                char c = getch();
                switch ( c ) {
                    case ' ':
                        g_Algorithm = (enum nbodyAlgorithm_enum) (g_Algorithm+1);
                        if ( g_Algorithm > g_maxAlgorithm ) {
                            g_Algorithm = g_bNoCPU ? GPU_AOS : CPU_AOS;
                        }
                        break;
                    case 'q':
                    case 'Q':
                        bStop = 1;
                        break;
                }

            }
        }
    }

    freeArrays();

    for ( int i = 0; i < g_numGPUs; i++ ) {
        struct gpuInit_struct initGPU = {i};
        worker_delegate(&g_GPUThreadPool[i], teardownGPU, &initGPU, 1);
        if ( cudaSuccess != initGPU.status ) {
            fprintf( stderr, "GPU %d teardown failed "
                " with %d (%s)\n",
                i,
                initGPU.status,
                cudaGetErrorString( initGPU.status ) );
            return 1;
        }
    }

    for (int i = 0; i < g_numGPUs; i++) {
        worker_delegate(&g_GPUThreadPool[i], NULL, NULL, 0);
        worker_join(&g_GPUThreadPool[i]);
        worker_destroy(&g_GPUThreadPool[i]);
    }

    return 0;
Error:
    if ( cudaSuccess != status ) {
        printf( "CUDA Error: %s\n", cudaGetErrorString( status ) );
    }
    return 1;
}

/* vim: set ts=4 sts=4 sw=4 et: */
