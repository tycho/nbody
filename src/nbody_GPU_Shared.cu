/*
 *
 * nbody_GPU_Shared.cu
 *
 * Shared memory-based implementation of the O(N^2) N-body calculation.
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
#include "nbody_util.h"
#include "nbody_GPU_Shared.h"
#include "bodybodyInteraction.cuh"

__global__ void
ComputeNBodyGravitation_Shared(
    float *force,
    float *posMass,
    float softeningSquared,
    size_t N )
{
    HIP_DYNAMIC_SHARED( float4, shPosMass)
    for ( size_t i = blockIdx.x*blockDim.x + threadIdx.x;
                 i < N;
                 i += blockDim.x*gridDim.x )
    {
        float acc[3] = {0};
        float4 myPosMass = ((float4 *) posMass)[i];
#pragma unroll 32
        for ( size_t j = 0; j < N; j += blockDim.x ) {
            shPosMass[threadIdx.x] = ((float4 *) posMass)[j+threadIdx.x];
            __syncthreads();
            for ( size_t k = 0; k < blockDim.x; k++ ) {
                float fx, fy, fz;
                float4 bodyPosMass = shPosMass[k];

                bodyBodyInteraction(
                    &fx, &fy, &fz,
                    myPosMass.x, myPosMass.y, myPosMass.z,
                    bodyPosMass.x,
                    bodyPosMass.y,
                    bodyPosMass.z,
                    bodyPosMass.w,
                    softeningSquared );
                acc[0] += fx;
                acc[1] += fy;
                acc[2] += fz;
            }
            __syncthreads();
        }
        force[4*i+0] = acc[0];
        force[4*i+1] = acc[1];
        force[4*i+2] = acc[2];
    }
}

DEFINE_AOS(ComputeGravitation_GPU_Shared)
{
    hipError_t status;
    hipEvent_t evStart = 0, evStop = 0;
    float ms = 0.0;
    HIP_CHECK( hipEventCreate( &evStart ) );
    HIP_CHECK( hipEventCreate( &evStop ) );
    HIP_CHECK( hipEventRecord( evStart, NULL ) );
    hipLaunchKernelGGL((ComputeNBodyGravitation_Shared), dim3(300), dim3(256), 256*sizeof(float4), 0,
        force,
        posMass,
        softeningSquared,
        N );
    HIP_CHECK( hipEventRecord( evStop, NULL ) );
    HIP_CHECK( hipDeviceSynchronize() );
    HIP_CHECK( hipEventElapsedTime( &ms, evStart, evStop ) );
Error:
    HIP_CHECK( hipEventDestroy( evStop ) );
    HIP_CHECK( hipEventDestroy( evStart ) );
    return ms;
}

/* vim: set ts=4 sts=4 sw=4 et: */
