/*
 *
 * nbody_GPU_AOS.cu
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
#include "nbody_GPU_AOS.h"
#include "bodybodyInteraction.cuh"

__global__ void
ComputeNBodyGravitation_GPU_AOS(
    float *force,
    float *posMass,
    size_t N,
    float softeningSquared )
{
    for ( size_t i = blockIdx.x*blockDim.x + threadIdx.x;
                 i < N;
                 i += blockDim.x*gridDim.x )
    {
        float acc[3] = {0};
        float4 me = ((float4 *) posMass)[i];
        float myX = me.x;
        float myY = me.y;
        float myZ = me.z;
        for ( size_t j = 0; j < N; j++ ) {
            float4 body = ((float4 *) posMass)[j];
            float fx, fy, fz;
            bodyBodyInteraction(
                &fx, &fy, &fz,
                myX, myY, myZ,
                body.x, body.y, body.z, body.w,
                softeningSquared);
            acc[0] += fx;
            acc[1] += fy;
            acc[2] += fz;
        }
        force[4*i+0] = acc[0];
        force[4*i+1] = acc[1];
        force[4*i+2] = acc[2];
    }
}

DEFINE_AOS(ComputeGravitation_GPU_AOS)
{
    hipError_t status;
    hipEvent_t evStart = 0, evStop = 0;
    float ms = 0.0;
    HIP_CHECK( hipEventCreate( &evStart ) );
    HIP_CHECK( hipEventCreate( &evStop ) );
    HIP_CHECK( hipEventRecord( evStart, NULL ) );
    hipLaunchKernelGGL((ComputeNBodyGravitation_GPU_AOS), dim3(300), dim3(256), 0, 0, 
        force, posMass, N, softeningSquared );
    HIP_CHECK( hipEventRecord( evStop, NULL ) );
    HIP_CHECK( hipDeviceSynchronize() );
    HIP_CHECK( hipEventElapsedTime( &ms, evStart, evStop ) );
Error:
    hipEventDestroy( evStop );
    hipEventDestroy( evStart );
    return ms;
}

/* vim: set ts=4 sts=4 sw=4 et: */
