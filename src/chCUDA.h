/*
 *
 * chCUDA.h
 *
 * Either loads CUDA or the dummy API interface, depending on build
 * requirements.
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

#pragma once

#ifdef USE_CUDA

#include "hip/hip_runtime.h"

#if CUDART_VERSION < 9000
#  define __shfl_sync(x,y,z) __shfl(y,z)
#  define __shfl_xor_sync(x,y,z) __shfl_xor(y,z)
#endif

#else

#include <stddef.h>
#include <math.h>
#include <memory.h>

#define __global__
#define __host__
#define __device__

typedef int hipError_t;
static const hipError_t hipSuccess = 0;

static inline hipError_t hipGetDeviceCount( int *p )
{
    if (!p)
        return 1;
    *p = 0;
    return 0;
}

static inline hipError_t hipMalloc ( void **devPtr, size_t size )
{
    return 1;
}

static inline hipError_t hipDeviceReset ( void )
{
    return 1;
}

static inline hipError_t hipHostMalloc ( void ** pHost, size_t size, unsigned int flags )
{
    return 1;
}

#define hipHostMallocMapped 0
#define hipHostMallocPortable 0

static inline hipError_t hipFree ( void * devPtr )
{
    return 1;
}

static inline hipError_t hipMemcpyAsync ( void * dst, const void * src, size_t count, int kind, int stream )
{
    return 1;
}

#define hipMemcpyHostToHost 0
#define hipMemcpyHostToDevice 0
#define hipMemcpyDeviceToHost 0
#define hipMemcpyDeviceToDevice 0
#define hipMemcpyDefault 0

struct hipDeviceProp_t
{
    int major;
    int minor;
};

static inline hipError_t hipGetDeviceProperties ( struct hipDeviceProp_t *  prop, int device )
{
    if (!prop)
        return 1;
    memset(prop, 0, sizeof(struct hipDeviceProp_t));
    return 0;
}

static inline hipError_t cudaSetDeviceFlags ( unsigned int flags )
{
    return 1;
}

#define cudaDeviceMapHost 0

static inline hipError_t hipSetDevice ( int device )
{
    return 1;
}

static inline float rsqrtf(float f)
{
    return 1.0f / sqrtf(f);
}

#endif
