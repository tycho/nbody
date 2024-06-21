/*
 *
 * nbody_util.cpp
 *
 * Common n-body utility functions.
 *
 * Copyright (c) 2019-2021, Uplink Laboratories, LLC
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

#include <stdint.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef HIGH_ENTROPY
#include <immintrin.h>
#endif
#ifdef _WIN32
#include <windows.h>
#else
#if !defined(__aarch64__) && !defined(_M_ARM64)
#include <mm_malloc.h>
#endif
#include <math.h>
#include <stdio.h>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/select.h>
#endif
#if defined(__arch64__) || defined(_M_ARM64)
#include <stdlib.h>
#define _mm_malloc(x,y) malloc(x)
#define _mm_free free
#endif
#include <random>

#include "nbody_util.h"

int processorCount(void)
{
#ifdef _OPENMP
    int k;
#  pragma omp parallel
    {
#  pragma omp master
        {
            k = omp_get_num_threads();
        }
    }
    return k;
#else
    return 1;
#endif
}

static std::linear_congruential_engine<std::uint32_t, 214013, 2531011, 1U << 31U> randNative;

void seedRandom(unsigned int seed)
{
    randNative.seed(seed);
}

float nbodyRandom(float randMin, float randMax)
{
    float result;
    uint32_t v;
    v = randNative();
    result = (float)v / (float)randNative.max();
    return ((1.0f - result) * randMin + result * randMax);
}

static inline float dot(float *v0, float *v1)
{
    return v0[0]*v1[0] + v0[1]*v1[1] + v0[2]*v1[2];
}

void randomUnitBodies(float *pos, float *vel, size_t N, float gscale, float velscale)
{
    const float scale = gscale * (N / 1024.0f);
    const float vscale = gscale * velscale;

    for ( size_t i = 0; i < N; i++ ) {
        float point[3];
        float velocity[3];
        float lenSqr;

        point[0] = nbodyRandom(-1.0f, 1.0f);
        point[1] = nbodyRandom(-1.0f, 1.0f);
        point[2] = nbodyRandom(-1.0f, 1.0f);

        lenSqr = dot(point, point);
        if (lenSqr > 1) {
            // Try again
            i--;
            continue;
        }

        velocity[0] = nbodyRandom(-1.0f, 1.0f);
        velocity[1] = nbodyRandom(-1.0f, 1.0f);
        velocity[2] = nbodyRandom(-1.0f, 1.0f);

        lenSqr = dot(velocity, velocity);
        if (lenSqr > 1) {
            // Try again
            i--;
            continue;
        }

        pos[4*i+0] = point[0] * scale;
        pos[4*i+1] = point[1] * scale;
        pos[4*i+2] = point[2] * scale;
        pos[4*i+3] = nbodyRandom(0.5f, 1.0f);

        vel[4*i+0] = velocity[0] * vscale;
        vel[4*i+1] = velocity[1] * vscale;
        vel[4*i+2] = velocity[2] * vscale;
        vel[4*i+3] = 1.0f / pos[4*i+3];
    }
}

void *alignedAlloc(size_t alignment, size_t size)
{
    return _mm_malloc(size, alignment);
}

void alignedFree(void *p)
{
    _mm_free(p);
}

#ifndef _WIN32
int kbhit(void)
{
    struct termios oldt, newt;
    int ch;
    int oldf;

    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

    ch = getchar();

    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    fcntl(STDIN_FILENO, F_SETFL, oldf);

    if(ch != EOF)
    {
        ungetc(ch, stdin);
        return 1;
    }

    return 0;
}

// we only call getch() when kbhit() has told us there
// is a pending keystroke
int getch(void)
{
    return getchar();
}
#endif
/* vim: set ts=4 sts=4 sw=4 et: */
