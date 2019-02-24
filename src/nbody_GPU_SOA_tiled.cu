/*
 *
 * nbody_GPU_SOA_tiled.h
 *
 * CUDA implementation of the O(N^2) N-body calculation.
 * Tiled to take advantage of the symmetry of gravitational
 * forces: Fij=-Fji
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
#include "nbody_GPU_SOA_tiled.h"
#include "bodybodyInteraction.cuh"

template<size_t nTile>
__device__ void
DoDiagonalTile_GPU_SOA(
    float *forceX, float *forceY, float *forceZ,
    float *posMass,
    float softeningSquared,
    size_t iTile, size_t jTile
)
{
    int laneid = threadIdx.x&0x1f;
    size_t i = iTile*nTile+laneid;
    float acc[3] = {0, 0, 0};
    float myX = posMass[i*4+0];
    float myY = posMass[i*4+1];
    float myZ = posMass[i*4+2];

    for ( size_t _j = 0; _j < nTile; _j++ ) {
        size_t j = jTile*nTile+_j;

        float fx, fy, fz;
        float4 body = ((float4 *) posMass)[j];

        bodyBodyInteraction(
            &fx, &fy, &fz,
            myX, myY, myZ,
            body.x, body.y, body.z, body.w,
            softeningSquared );
        acc[0] += fx;
        acc[1] += fy;
        acc[2] += fz;
    }

    atomicAdd( &forceX[i], acc[0] );
    atomicAdd( &forceY[i], acc[1] );
    atomicAdd( &forceZ[i], acc[2] );
}

inline float
__device__
warpReduce( float x )
{
    x += __int_as_float( __shfl_xor_sync( 0xFFFFFFFF, __float_as_int(x), 16 ) );
    x += __int_as_float( __shfl_xor_sync( 0xFFFFFFFF, __float_as_int(x),  8 ) );
    x += __int_as_float( __shfl_xor_sync( 0xFFFFFFFF, __float_as_int(x),  4 ) );
    x += __int_as_float( __shfl_xor_sync( 0xFFFFFFFF, __float_as_int(x),  2 ) );
    x += __int_as_float( __shfl_xor_sync( 0xFFFFFFFF, __float_as_int(x),  1 ) );
    return x;
}

template<size_t nTile>
__device__ void
DoNondiagonalTile_GPU_SOA(
    float *forceX, float *forceY, float *forceZ,
    float *posMass,
    float softeningSquared,
    size_t iTile, size_t jTile
)
{
    int laneid = threadIdx.x&0x1f;
    size_t i = iTile*nTile+laneid;
    float ax = 0.0f, ay = 0.0f, az = 0.0f;
    float4 myPosMass = ((float4 *) posMass)[i];
    float myX = myPosMass.x;
    float myY = myPosMass.y;
    float myZ = myPosMass.z;

    float4 shufSrcPosMass = ((float4 *) posMass)[jTile*nTile+laneid];

//#pragma unroll
    for ( size_t _j = 0; _j < nTile; _j++ ) {
        const size_t j = jTile*nTile+_j;

        float fx, fy, fz;
        float4 bodyPosMass;

        bodyPosMass.x = __shfl_sync( 0xFFFFFFFF, shufSrcPosMass.x, _j );
        bodyPosMass.y = __shfl_sync( 0xFFFFFFFF, shufSrcPosMass.y, _j );
        bodyPosMass.z = __shfl_sync( 0xFFFFFFFF, shufSrcPosMass.z, _j );
        bodyPosMass.w = __shfl_sync( 0xFFFFFFFF, shufSrcPosMass.w, _j );

        bodyBodyInteraction(
            &fx, &fy, &fz,
            myX, myY, myZ,
            bodyPosMass.x, bodyPosMass.y, bodyPosMass.z, bodyPosMass.w,
            softeningSquared );

        ax += fx;
        ay += fy;
        az += fz;

        fx = warpReduce( -fx );
        fy = warpReduce( -fy );
        fz = warpReduce( -fz );

        if ( laneid == 0 ) {
            atomicAdd( &forceX[j], fx );
            atomicAdd( &forceY[j], fy );
            atomicAdd( &forceZ[j], fz );
        }
    }

    atomicAdd( &forceX[i], ax );
    atomicAdd( &forceY[i], ay );
    atomicAdd( &forceZ[i], az );

}

template<size_t nTile>
__global__ void
ComputeNBodyGravitation_GPU_SOA_tiled(
    float *forceX, float *forceY, float *forceZ,
    float *posMass,
    size_t N,
    float softeningSquared )
{
    int warpsPerBlock = nTile/32;
    const int warpid = threadIdx.x >> 5;

    int iTileCoarse = blockIdx.x;
    int iTile = iTileCoarse*warpsPerBlock+warpid;
    int jTile = blockIdx.y;

    if ( iTile == jTile ) {
        DoDiagonalTile_GPU_SOA<32>( forceX, forceY, forceZ, posMass, softeningSquared, iTile, jTile );
    }
    else if ( jTile < iTile ) {
        DoNondiagonalTile_GPU_SOA<32>( forceX, forceY, forceZ, posMass, softeningSquared, iTile, jTile );
    }
}

template<size_t nTile>
hipError_t
ComputeGravitation_GPU_SOA_tiled(
    float *forces[3],
    float *posMass,
    float softeningSquared,
    size_t N
)
{
    hipError_t status;
    dim3 blocks( N/nTile, N/32, 1 );

    HIP_CHECK( hipMemset( forces[0], 0, N*sizeof(float) ) );
    HIP_CHECK( hipMemset( forces[1], 0, N*sizeof(float) ) );
    HIP_CHECK( hipMemset( forces[2], 0, N*sizeof(float) ) );
    hipLaunchKernelGGL((ComputeNBodyGravitation_GPU_SOA_tiled<nTile>), dim3(blocks), dim3(nTile), 0, 0,  forces[0], forces[1], forces[2], posMass, N, softeningSquared );
    HIP_CHECK( hipDeviceSynchronize() );
Error:
    return status;
}

__global__ void
AOStoSOA_GPU_3( float *outX, float *outY, float *outZ, const float *in, size_t N )
{
    for ( size_t i = blockIdx.x*blockDim.x + threadIdx.x;
                 i < N;
                 i += blockDim.x*gridDim.x ) {
        float tmp[3] = { in[4*i+0], in[4*i+1], in[4*i+2] };
        outX[i] = tmp[0];
        outY[i] = tmp[1];
        outZ[i] = tmp[2];
    }
}

__global__ void
SOAtoAOS_GPU_3( float *out, const float *inX, const float *inY, const float *inZ, size_t N )
{
    for ( size_t i = blockIdx.x*blockDim.x + threadIdx.x;
                 i < N;
                 i += blockDim.x*gridDim.x ) {
        out[4*i+0] = inX[i];
        out[4*i+1] = inY[i];
        out[4*i+2] = inZ[i];
    }
}

DEFINE_AOS(ComputeGravitation_GPU_SOA_tiled)
{
    hipError_t status;
    hipEvent_t evStart = 0, evStop = 0;
    float ms = 0.0;

    float *forces[3] = {0};
    HIP_CHECK( hipMalloc( &forces[0], N*sizeof(float) ) );
    HIP_CHECK( hipMalloc( &forces[1], N*sizeof(float) ) );
    HIP_CHECK( hipMalloc( &forces[2], N*sizeof(float) ) );

    HIP_CHECK( hipEventCreate( &evStart ) );
    HIP_CHECK( hipEventCreate( &evStop ) );

    hipLaunchKernelGGL((AOStoSOA_GPU_3), dim3(300), dim3(256), 0, 0,  forces[0], forces[1], forces[2], force, N );

    HIP_CHECK( hipEventRecord( evStart, NULL ) );
    HIP_CHECK( ComputeGravitation_GPU_SOA_tiled<128>(
        forces,
        posMass,
        softeningSquared,
        N ) );
    HIP_CHECK( hipEventRecord( evStop, NULL ) );

    HIP_CHECK( hipDeviceSynchronize() );
    hipLaunchKernelGGL((SOAtoAOS_GPU_3), dim3(300), dim3(256), 0, 0,  force, forces[0], forces[1], forces[2], N );

    HIP_CHECK( hipDeviceSynchronize() );
    HIP_CHECK( hipEventElapsedTime( &ms, evStart, evStop ) );
Error:
    HIP_CHECK( hipEventDestroy( evStop ) );
    HIP_CHECK( hipEventDestroy( evStart ) );
    return ms;
}

/* vim: set ts=4 sts=4 sw=4 et: */
