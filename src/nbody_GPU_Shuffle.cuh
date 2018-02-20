/*
 *
 * nbody_GPU_Shuffle.h
 *
 * Warp shuffle-based implementation of the O(N^2) N-body calculation.
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

#include <cuda_runtime_api.h>

#if CUDART_VERSION < 9000
#  define __shfl_sync(x,y,z) __shfl(y,z)
#endif

__global__ void
ComputeNBodyGravitation_Shuffle(
    float *force,
    float *posMass,
    float softeningSquared,
    size_t N )
{
    const int laneid = threadIdx.x & 31;
    for ( size_t i = blockIdx.x*blockDim.x + threadIdx.x;
                 i < N;
                 i += blockDim.x*gridDim.x )
    {
        float acc[3] = {0};
        float4 myPosMass = ((float4 *) posMass)[i];

        for ( size_t j = 0; j < N; j += 32 ) {
            float4 shufSrcPosMass = ((float4 *) posMass)[j+laneid];
#pragma unroll 32
            for ( size_t k = 0; k < 32; k++ ) {
                float fx, fy, fz;
                float4 shufDstPosMass;

                shufDstPosMass.x = __shfl_sync( 0xFFFFFFFF, shufSrcPosMass.x, k );
                shufDstPosMass.y = __shfl_sync( 0xFFFFFFFF, shufSrcPosMass.y, k );
                shufDstPosMass.z = __shfl_sync( 0xFFFFFFFF, shufSrcPosMass.z, k );
                shufDstPosMass.w = __shfl_sync( 0xFFFFFFFF, shufSrcPosMass.w, k );

                bodyBodyInteraction(
                    &fx, &fy, &fz,
                    myPosMass.x, myPosMass.y, myPosMass.z,
                    shufDstPosMass.x,
                    shufDstPosMass.y,
                    shufDstPosMass.z,
                    shufDstPosMass.w,
                    softeningSquared);
                acc[0] += fx;
                acc[1] += fy;
                acc[2] += fz;
            }
        }

        force[4*i+0] = acc[0];
        force[4*i+1] = acc[1];
        force[4*i+2] = acc[2];
    }
}

float
ComputeGravitation_GPU_Shuffle( float *force, float *posMass, float softeningSquared, size_t N )
{
    cudaError_t status;
    cudaEvent_t evStart = 0, evStop = 0;
    float ms = 0.0f;
    CUDART_CHECK( cudaEventCreate( &evStart ) );
    CUDART_CHECK( cudaEventCreate( &evStop ) );
    CUDART_CHECK( cudaEventRecord( evStart, NULL ) );
    ComputeNBodyGravitation_Shuffle <<<300,256>>>( force, posMass, softeningSquared, N );
    CUDART_CHECK( cudaEventRecord( evStop, NULL ) );
    CUDART_CHECK( cudaDeviceSynchronize() );
    CUDART_CHECK( cudaEventElapsedTime( &ms, evStart, evStop ) );
Error:
    CUDART_CHECK( cudaEventDestroy( evStop ) );
    CUDART_CHECK( cudaEventDestroy( evStart ) );
    return ms;
}

/* vim: set ts=4 sts=4 sw=4 et: */
