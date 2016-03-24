/*
 *
 * nbody_util.h
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

#ifndef __included_nbody_util_h
#define __included_nbody_util_h

#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NBODY_ALIGNMENT 64

#if defined(__GNUC__)
#  define ALIGNED(n) __attribute__((aligned(n)))
#  define ASSERT_ALIGNED(p,n) do { assert(((uintptr_t)(p) & (uintptr_t)(n-1)) == 0); } while (0)
#  define ASSUME(cond) do { if (!(cond)) __builtin_unreachable(); } while (0)
#else
#  define ALIGNED(n)
#  define ASSERT_ALIGNED(p,n)
#  define ASSUME(cond)
#endif

extern const char *rgszAlgorithmNames[];
int processorCount(void);

// "unaligned" float, or more accurately a float with no alignment guarantees.
typedef float ufloat;

// aligned float
typedef ufloat ALIGNED(NBODY_ALIGNMENT) afloat;

#ifdef __cplusplus
}
#endif

#endif
