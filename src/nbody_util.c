/*
 *
 * nbody.cu
 *
 * N-body example that illustrates gravitational simulation.
 * This is the type of computation that GPUs excel at:
 * parallelizable, with lots of FLOPS per unit of external
 * memory bandwidth required.
 *
 * Requires: No minimum SM requirement.  If SM 3.x is not available,
 * this application quietly replaces the shuffle and fast-atomic
 * implementations with the shared memory implementation.
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
#include <malloc.h>
#include <math.h>
#include <stdio.h>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/select.h>
#endif

#include "nbody_util.h"

#ifdef _WIN32
/*
this code uses the same lagged fibonacci generator as the
original bsd random implementation except for the seeding
which was broken in the original
*/

static uint32_t init[] = {
0x00000000,0x5851f42d,0xc0b18ccf,0xcbb5f646,
0xc7033129,0x30705b04,0x20fd5db4,0x9a8b7f78,
0x502959d8,0xab894868,0x6c0356a7,0x88cdb7ff,
0xb477d43f,0x70a3a52b,0xa8e4baf1,0xfd8341fc,
0x8ae16fd9,0x742d2f7a,0x0d1f0796,0x76035e09,
0x40f7702c,0x6fa72ca5,0xaaa84157,0x58a0df74,
0xc74a0364,0xae533cc4,0x04185faf,0x6de3b115,
0x0cab8628,0xf043bfa4,0x398150e9,0x37521657};

static int n = 31;
static int i = 3;
static int j = 0;
static uint32_t *x = init+1;
static volatile int lock[2];

static uint32_t lcg31(uint32_t x) {
    return (1103515245*x + 12345) & 0x7fffffff;
}

static uint64_t lcg64(uint64_t x) {
    return 6364136223846793005ull*x + 1;
}

static void srandom(unsigned seed) {
    int k;
    uint64_t s = seed;

    if (n == 0) {
        x[0] = s;
        return;
    }
    i = n == 31 || n == 7 ? 3 : 1;
    j = 0;
    for (k = 0; k < n; k++) {
        s = lcg64(s);
        x[k] = s>>32;
    }
    /* make sure x contains at least one odd number */
    x[0] |= 1;
}

static long random(void) {
    long k;

    if (n == 0) {
        k = x[0] = lcg31(x[0]);
        return k;
    }
    x[i] += x[j];
    k = x[i]>>1;
    if (++i == n)
        i = 0;
    if (++j == n)
        j = 0;
    return k;
}
#endif

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

void seedRandom(unsigned int seed)
{
    srandom(seed);
}

float nbodyRandom(float randMin, float randMax)
{
    float result;
    uint32_t v;
#if defined(HIGH_ENTROPY) && defined __RDRND__
    int i = _rdrand32_step(&v);
    if (!i)
        abort();
#else
    v = random();
#endif
    result = (float)v / (float)RAND_MAX;
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
        pos[4*i+3] = 1.0f;

        vel[4*i+0] = velocity[0] * vscale;
        vel[4*i+1] = velocity[1] * vscale;
        vel[4*i+2] = velocity[2] * vscale;
        vel[4*i+3] = 1.0f;
    }
}

void *alignedAlloc(size_t alignment, size_t size)
{
#ifdef _WIN32
    return VirtualAlloc(0, size, MEM_RESERVE|MEM_COMMIT, PAGE_READWRITE);
#else
    return memalign(alignment, size);
#endif
}

void alignedFree(void *p)
{
#ifdef _WIN32
    VirtualFree(p, 0, MEM_RELEASE);
#else
    free(p);
#endif
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
