/*
 *
 * nbody.h
 *
 * Header file to declare globals in nbody.cu
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

#ifndef __NBODY_H__
#define __NBODY_H__

#include "nbody_util.h"
#include "nbody_CPU_SIMD.h"

extern float *g_hostAOS_PosMass;
extern float *g_hostAOS_VelInvMass;
extern float *g_hostAOS_Force;

extern float *g_hostSOA_Pos[3];
extern float *g_hostSOA_Force[3];
extern float *g_hostSOA_Mass;
extern float *g_hostSOA_InvMass;

extern int g_numCPUCores;
extern int g_numGPUs;

extern float ComputeGravitation_multiGPU  ( float *force, float *posMass, float softeningSquared, size_t N );

typedef enum {
    ALGORITHM_NONE,
    ALGORITHM_SOA,
    ALGORITHM_AOS,
    ALGORITHM_AOS_GPU,
    ALGORITHM_AOS_MGPU,
} algorithm_t;

// There are two different function prototypes for ComputeGravitation,
// depending on whether the algorithm uses the SOA or AOS structures.
typedef float (*pfnComputeGravitation_AOS_t)( float * restrict force, float * restrict posMass, float softeningSquared, size_t N);
typedef float (*pfnComputeGravitation_SOA_t)( float ** restrict force, float ** restrict pos, float * restrict mass, float softeningSquared, size_t N);

typedef struct _algorithm_def_t {
    const char *name;
    algorithm_t type;
    union {
        pfnComputeGravitation_AOS_t aos;
        pfnComputeGravitation_SOA_t soa;
    };
} algorithm_def_t, *palgorithm_def_t;

#ifdef HAVE_SIMD
#  ifdef _MSC_VER
#    ifdef _M_X64
#      ifdef __AVX__
#        define HAVE_AVX
#      else
#        define HAVE_SSE
#      endif
#    else
#      error "SSE/AVX intrinsics are unsupported for 32-bit targets."
#    endif
#  else
#    if defined(__SSE__) && !defined(__AVX__)
#      define HAVE_SSE
#    elif defined(__AVX__)
#      define HAVE_AVX
#    endif
#  endif
#endif


#endif

/* vim: set ts=4 sts=4 sw=4 et: */
