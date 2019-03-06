/*
 *
 * nbody_GPU_AOS_tiled_const.cu
 *
 * CUDA implementation of the O(N^2) N-body calculation.
 * Tiled to take advantage of the symmetry of gravitational
 * forces: Fij=-Fji
 * Uses __constant__ device memory to hold bodies, and does
 * multiple passes over the data for the O(N^2) calculations.
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
#include "nbody_GPU_AOS_tiled_const.h"
#include "bodybodyInteraction.cuh"

template<int nTile>
__device__ void
DoDiagonalTile_GPU_const(
    float *force,
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

    atomicAdd( &force[4*i+0], acc[0] );
    atomicAdd( &force[4*i+1], acc[1] );
    atomicAdd( &force[4*i+2], acc[2] );
}

inline float
__device__
warpReduce_const( float x )
{
    x += __int_as_float( __shfl_xor_sync( 0xFFFFFFFF, __float_as_int(x), 16 ) );
    x += __int_as_float( __shfl_xor_sync( 0xFFFFFFFF, __float_as_int(x),  8 ) );
    x += __int_as_float( __shfl_xor_sync( 0xFFFFFFFF, __float_as_int(x),  4 ) );
    x += __int_as_float( __shfl_xor_sync( 0xFFFFFFFF, __float_as_int(x),  2 ) );
    x += __int_as_float( __shfl_xor_sync( 0xFFFFFFFF, __float_as_int(x),  1 ) );
    return x;
}

template<int nTile>
__device__ void
DoNondiagonalTile_GPU_const(
    float *force,
    float *posMass,
    float softeningSquared,
    size_t iTile, size_t jTile,
    volatile float *sForces
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

    for ( size_t _j = 0; _j < nTile; _j++ ) {
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

        sForces[0*1056+33*laneid+_j] = fx;
        sForces[1*1056+33*laneid+_j] = fy;
        sForces[2*1056+33*laneid+_j] = fz;
    }

    atomicAdd( &force[4*i+0], ax );
    atomicAdd( &force[4*i+1], ay );
    atomicAdd( &force[4*i+2], az );

    {
        size_t j = jTile*nTile+laneid;

        ax = 0.0f;
#pragma unroll 32
        for ( int _j = 0; _j < nTile; _j++ ) {
            ax -= sForces[0*1056+33*_j+laneid];
        }
        atomicAdd( &force[4*j+0], ax );

        ay = 0.0f;
#pragma unroll 32
        for ( int _j = 0; _j < nTile; _j++ ) {
            ay -= sForces[1*1056+33*_j+laneid];
        }
        atomicAdd( &force[4*j+1], ay );

        az = 0.0f;
#pragma unroll 32
        for ( int _j = 0; _j < nTile; _j++ ) {
            az -= sForces[2*1056+33*_j+laneid];
        }
        atomicAdd( &force[4*j+2], az );
    }

}


template<int nTile>
__global__ void
ComputeNBodyGravitation_GPU_tiled_const(
    float *force,
    float *posMass,
    size_t N,
    float softeningSquared )
{
    int warpsPerBlock = nTile/32;
    const int warpid = threadIdx.x >> 5;
    //
    // each 32x32 tile needs 3 float vectors (one fo reach dimension
    // of force, padded to 33 floats per row to avoid bank conflicts
    //
    // 3K floats = 12672 bytes of shared memory per 32x32 tile
    //
    __shared__ float sForces[3*33*32];

    int iTileCoarse = blockIdx.x;
    int iTile = iTileCoarse*warpsPerBlock+warpid;
    int jTile = blockIdx.y;

    if ( iTile == jTile ) {
        DoDiagonalTile_GPU_const<32>(
            force,
            posMass,
            softeningSquared,
            iTile, jTile );
    }
    else if ( jTile < iTile ) {
        DoNondiagonalTile_GPU_const<32>(
            force,
            posMass,
            softeningSquared,
            iTile, jTile, sForces /*+warpid*(3*33*32)*/ );
    }
}

template<int nTile>
cudaError_t
ComputeGravitation_GPU_AOS_tiled_const(
    float *force,
    float *posMass,
    float softeningSquared,
    size_t N
)
{
    cudaError_t status;
    dim3 blocks( N/nTile, N/32, 1 );

    CUDART_CHECK( cudaMemset( force, 0, 4*N*sizeof(float) ) );
    ComputeNBodyGravitation_GPU_tiled_const<nTile><<<blocks,nTile>>>( force, posMass, N, softeningSquared );
    CUDART_CHECK( cudaDeviceSynchronize() );
Error:
    return status;
}

DEFINE_AOS(ComputeGravitation_GPU_AOS_tiled_const)
{
    cudaError_t status;
    cudaEvent_t evStart = 0, evStop = 0;
    float ms = 0.0;
    CUDART_CHECK( cudaDeviceSetCacheConfig( cudaFuncCachePreferShared ) );
    CUDART_CHECK( cudaEventCreate( &evStart ) );
    CUDART_CHECK( cudaEventCreate( &evStop ) );
    CUDART_CHECK( cudaEventRecord( evStart, NULL ) );
    CUDART_CHECK( ComputeGravitation_GPU_AOS_tiled_const<32>(
        force,
        posMass,
        softeningSquared,
        N ) );
    CUDART_CHECK( cudaEventRecord( evStop, NULL ) );
    CUDART_CHECK( cudaDeviceSynchronize() );
    CUDART_CHECK( cudaEventElapsedTime( &ms, evStart, evStop ) );
Error:
    CUDART_CHECK( cudaEventDestroy( evStop ) );
    CUDART_CHECK( cudaEventDestroy( evStart ) );
    return ms;
}

/* vim: set ts=4 sts=4 sw=4 et: */