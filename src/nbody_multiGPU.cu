/*
 *
 * nbody_multiGPU.cu
 *
 * Single-threaded multi-GPU implementation of the O(N^2) N-body calculation.
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

#include "hip/hip_runtime.h"

#include "chError.h"

#include "nbody.h"
#include "nbody_multiGPU_shared.cuh"

#include "bodybodyInteraction.cuh"

// maximum number of GPUs supported by single-threaded multi-GPU
const int g_maxGPUs = 32;

__global__ void
ComputeNBodyGravitation_multiGPU(
    float *force,
    float *posMass,
    float softeningSquared,
    size_t base,
    size_t n,
    size_t N )
{
    ComputeNBodyGravitation_Shared_multiGPU(
        force,
        posMass,
        softeningSquared,
        base,
        n,
        N );
}

float
ComputeGravitation_multiGPU(
    float *force,
    float *posMass,
    float softeningSquared,
    size_t N
)
{
    hipError_t status;

    hipEvent_t evStart = 0, evStop = 0;
    float ms = 0.0;

    float *dptrPosMass[g_maxGPUs];
    float *dptrForce[g_maxGPUs];
    int oldDevice;

    memset( dptrPosMass, 0, sizeof(dptrPosMass) );
    memset( dptrForce, 0, sizeof(dptrForce) );
    size_t bodiesPerGPU = N / g_numGPUs;
    if ( (0 != N % g_numGPUs) || (g_numGPUs > g_maxGPUs) ) {
        return 0.0f;
    }
    HIP_CHECK( hipGetDevice( &oldDevice ) );

    HIP_CHECK( hipSetDevice( 0 ) );
    HIP_CHECK( hipEventCreate( &evStart ) );
    HIP_CHECK( hipEventCreate( &evStop ) );
    HIP_CHECK( hipEventRecord( evStart, NULL ) );

    // kick off the asynchronous memcpy's - overlap GPUs pulling
    // host memory with the CPU time needed to do the memory
    // allocations.
    for ( size_t i = 0; i < g_numGPUs; i++ ) {
        HIP_CHECK( hipSetDevice( i ) );
        HIP_CHECK( hipMalloc( &dptrPosMass[i], 4*N*sizeof(float) ) );
        HIP_CHECK( hipMalloc( &dptrForce[i], 4*bodiesPerGPU*sizeof(float) ) );
        HIP_CHECK( hipMemcpyAsync(
            dptrPosMass[i],
            g_hostAOS_PosMass,
            4*N*sizeof(float),
            hipMemcpyHostToDevice ) );
    }
    for ( size_t i = 0; i < g_numGPUs; i++ ) {
        HIP_CHECK( hipSetDevice( i ) );
        hipLaunchKernelGGL((ComputeNBodyGravitation_multiGPU), dim3(300), dim3(256), 256*sizeof(float4), 0, 
            dptrForce[i],
            dptrPosMass[i],
            softeningSquared,
            i*bodiesPerGPU,
            bodiesPerGPU,
            N );
        HIP_CHECK( hipMemcpyAsync(
            g_hostAOS_Force+4*bodiesPerGPU*i,
            dptrForce[i],
            4*bodiesPerGPU*sizeof(float),
            hipMemcpyDeviceToHost ) );
    }

    HIP_CHECK( hipSetDevice( 0 ) );
    HIP_CHECK( hipEventRecord( evStop, NULL ) );

    // Synchronize with each GPU in turn.
    for ( size_t i = 0; i < g_numGPUs; i++ ) {
        HIP_CHECK( hipSetDevice( i ) );
        HIP_CHECK( hipDeviceSynchronize() );
    }

    HIP_CHECK( hipEventElapsedTime( &ms, evStart, evStop ) );
Error:
    for ( size_t i = 0; i < g_numGPUs; i++ ) {
        hipFree( dptrPosMass[i] );
        hipFree( dptrForce[i] );
    }
    hipSetDevice( oldDevice );
    return ms;
}

/* vim: set ts=4 sts=4 sw=4 et: */
