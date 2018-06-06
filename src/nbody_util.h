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
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NBODY_ALIGNMENT 64

/* Disabled until GCC 9, when some of the target_clones bugs are hopefully fixed. */
#if !defined(__GNUC__) || __GNUC__ < 9
#define TARGET_DECL
#endif

#if defined(__GNUC__)
#  define ALIGNED(n) __attribute__((aligned(n)))
#  ifdef DEBUG
#    define ASSERT_ALIGNED(p,n) do { assert(((uintptr_t)(p) & (uintptr_t)(n-1)) == 0); } while (0)
#  else
#    define ASSERT_ALIGNED(p,n)
#  endif
#  if defined(__clang__)
#    if __has_builtin(__builtin_assume)
#      define ASSUME(cond) __builtin_assume(cond)
#    endif
#  endif
#  if !defined(ASSUME) && defined(__GNUC__) && (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 5))
#    define ASSUME(cond) do { if (!(cond)) __builtin_unreachable(); } while (0)
#  endif
#  if !defined(__clang__) && !defined(TARGET_DECL)
#    define TARGET_DECL __attribute__((target_clones("arch=haswell", "arch=sandybridge", "default")))
#  endif
#endif

#if !defined(ALIGNED)
#  define ALIGNED(n)
#endif
#if !defined(ASSERT_ALIGNED)
#  define ASSERT_ALIGNED(p,n)
#endif
#if !defined(ASSUME)
#  define ASSUME(cond)
#endif
#if !defined(TARGET_DECL)
#  define TARGET_DECL
#endif

#define DECLARE_SOA(Name) \
	float \
	Name( \
		afloat ** restrict force, \
		afloat ** restrict pos, \
		afloat *  restrict mass, \
		float softeningSquared, \
		size_t N )

#define DECLARE_AOS(Name) \
	float \
	Name( \
		afloat * restrict force, \
		afloat * restrict posMass, \
		float softeningSquared, \
		size_t N )

#define DEFINE_SOA(Name) \
	TARGET_DECL \
	static DECLARE_SOA(_ ## Name); \
	DECLARE_SOA(Name) { return _ ## Name(force, pos, mass, softeningSquared, N); } \
	TARGET_DECL \
	static DECLARE_SOA(_ ## Name)

#define DEFINE_AOS(Name) \
	TARGET_DECL \
	static DECLARE_AOS(_ ## Name); \
	DECLARE_AOS(Name) { return _ ## Name(force, posMass, softeningSquared, N); } \
	TARGET_DECL \
	static DECLARE_AOS(_ ## Name)


int processorCount(void);

void seedRandom(unsigned int seed);
float nbodyRandom(float randMin, float randMax);

void randomUnitBodies(float *pos, float *vel, size_t N, float scale, float velscale);

void *alignedAlloc(size_t alignment, size_t size);
void alignedFree(void *p);

#ifndef _WIN32
int kbhit(void);
int getch(void);
#endif

// "unaligned" float, or more accurately a float with no alignment guarantees.
typedef float ufloat;

// aligned float
typedef ufloat ALIGNED(NBODY_ALIGNMENT) afloat;

#ifdef __cplusplus
}
#endif

#endif
