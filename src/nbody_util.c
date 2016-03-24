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
#ifdef _WIN32
#include <windows.h>
#else
#include <malloc.h>
#include <stdio.h>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/select.h>
#endif

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

static inline void randomVector(float *v, size_t offset)
{
    v[offset+0] = nbodyRandom( 3.0f, 50.0f );
    v[offset+1] = nbodyRandom( 3.0f, 50.0f );
    v[offset+2] = nbodyRandom( 3.0f, 50.0f );
}

void randomUnitBodies(float *pos, float *vel, size_t N)
{
    for ( size_t i = 0; i < N; i++ ) {
        randomVector( pos, 4 * i );
        randomVector( vel, 4 * i );
        pos[4*i+3] = nbodyRandom( 1.0f, 1000.0f );  // unit mass
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
