/*
 *
 * nbody_GPU_Atomic.cu
 *
 * CUDA implementation of the O(N^2) N-body calculation.
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

#include "chError.h"
#include "nbody_util.h"
#include "nbody_GPU_Atomic.h"
#include "bodybodyInteraction.cuh"

//
// Atomics only make sense for SM 3.x and higher
//
template<typename T>
__global__ void
ComputeNBodyGravitation_Atomic( T *force, T *posMass, size_t N, T softeningSquared )
{
    for ( size_t i = blockIdx.x*blockDim.x + threadIdx.x;
                 i < N;
                 i += blockDim.x*gridDim.x )
    {
        float4 me = ((float4 *) posMass)[i];
        T acc[3] = {0.0f, 0.0f, 0.0f};
        T myX = me.x;
        T myY = me.y;
        T myZ = me.z;
        for ( size_t j = 0; j < i; j++ ) {
            float4 body = ((float4 *) posMass)[j];

            T fx, fy, fz;
            bodyBodyInteraction(
                &fx, &fy, &fz,
                myX, myY, myZ,
                body.x, body.y, body.z, body.w,
                softeningSquared );

            acc[0] += fx;
            acc[1] += fy;
            acc[2] += fz;

            float *f = &force[4*j+0];
            atomicAdd( f+0, -fx );
            atomicAdd( f+1, -fy );
            atomicAdd( f+2, -fz );

        }

        atomicAdd( &force[4*i+0], acc[0] );
        atomicAdd( &force[4*i+1], acc[1] );
        atomicAdd( &force[4*i+2], acc[2] );
    }
}

DEFINE_AOS(ComputeGravitation_GPU_Atomic)
{
    hipError_t status;
    hipEvent_t evStart = 0, evStop = 0;
    float ms = 0.0;
    HIP_CHECK( hipEventCreate( &evStart ) );
    HIP_CHECK( hipEventCreate( &evStop ) );
    HIP_CHECK( hipEventRecord( evStart, NULL ) );
    HIP_CHECK( hipMemset( force, 0, 4*N*sizeof(float) ) );
    hipLaunchKernelGGL((ComputeNBodyGravitation_Atomic<float>), dim3(300), dim3(256), 0, 0,  force, posMass, N, softeningSquared );
    HIP_CHECK( hipEventRecord( evStop, NULL ) );
    HIP_CHECK( hipDeviceSynchronize() );
    HIP_CHECK( hipEventElapsedTime( &ms, evStart, evStop ) );
Error:
    HIP_CHECK( hipEventDestroy( evStop ) );
    HIP_CHECK( hipEventDestroy( evStart ) );
    return ms;
}

/* vim: set ts=4 sts=4 sw=4 et: */
