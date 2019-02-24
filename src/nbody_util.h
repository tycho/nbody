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

#pragma once

#include <assert.h>
#include <cstdlib>

#ifdef __cplusplus
extern "C" {
#endif

#define NBODY_ALIGNMENT 64

/* Disabled until the target_clones bugs are fixed. */
//#if !defined(__GNUC__) || __GNUC__ < 9
#define TARGET_DECL
//#endif

#ifndef __has_builtin         // Optional of course.
#  define __has_builtin(x) 0  // Compatibility with non-clang compilers.
#endif

#if defined(__GNUC__)
#  define NOINLINE __attribute__((noinline))
#elif defined(_MSC_VER)
#  define NOINLINE __declspec(noinline)
#else
#  define NOINLINE
#endif

#if defined(__GNUC__)
#  define ALIGNED(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER)
#  define ALIGNED(n) __declspec(align(n))
#else
#  define ALIGNED(n)
#endif

#if __has_builtin(__builtin_assume)
#  define ASSUME(cond) __builtin_assume(cond)
#elif defined(__GNUC__) && (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 5))
#  define ASSUME(cond) do { if (!(cond)) __builtin_unreachable(); } while (0)
#else
#  define ASSUME(cond)
#endif

#if defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER) && !defined(__NVCC__) && !defined(__CUDA__)
#  define NO_FAST_MATH __attribute__((optimize("-fno-fast-math")))
#else
#  define NO_FAST_MATH
#endif

#if __has_builtin(__builtin_assume_aligned)
#  define ASSERT_ALIGNED(p,n) do { p = (float *)__builtin_assume_aligned(p, n); } while(0)
#else
#  define ASSERT_ALIGNED(p,n)
#endif

#if !defined(TARGET_DECL) && defined(__GNUC__) && !defined(__clang__)
#  define TARGET_DECL __attribute__((target_clones("arch=haswell", "arch=sandybridge", "default")))
#else
#  define TARGET_DECL
#endif

#define DECLARE_SOA(Name) \
	float \
	Name( \
		float ** __restrict force, \
		float ** __restrict pos, \
		float *  __restrict mass, \
		float softeningSquared, \
		size_t N )

#define DECLARE_AOS(Name) \
	float \
	Name( \
		float * __restrict force, \
		float * __restrict posMass, \
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

#ifdef __cplusplus
}
#endif
