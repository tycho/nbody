/*
 *
 * nbody_CPU_AVX.cpp
 *
 * Multithreaded AVX CPU implementation of the O(N^2) N-body calculation.
 * Uses SOA (structure of arrays) representation because it is a much
 * better fit for AVX.
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

#include "nbody.h"

#if defined(HAVE_AVX)
#include "libtime.h"

#include "bodybodyInteraction_AVX.h"
#include "nbody_CPU_SIMD.h"

float
ComputeGravitation_SIMD(
    float ** restrict force,
    float ** restrict pos,
    float *  restrict mass,
    float softeningSquared,
    size_t N
)
{
    uint64_t start, end;

    start = libtime_cpu();

    #pragma omp parallel for schedule(guided, 16)
    #pragma vector aligned
    #pragma ivdep
    for ( size_t i = 0; i < N; i++ )
    {
        __m256 ax = _mm256_setzero_ps();
        __m256 ay = _mm256_setzero_ps();
        __m256 az = _mm256_setzero_ps();
        __m256 *px = (__m256 *) pos[0];
        __m256 *py = (__m256 *) pos[1];
        __m256 *pz = (__m256 *) pos[2];
        __m256 *pmass = (__m256 *) mass;
        __m256 x0 = _mm256_set1_ps( pos[0][i] );
        __m256 y0 = _mm256_set1_ps( pos[1][i] );
        __m256 z0 = _mm256_set1_ps( pos[2][i] );

        #pragma vector aligned
        #pragma ivdep
        for ( size_t j = 0; j < N/8; j++ ) {

            bodyBodyInteraction(
                &ax, &ay, &az,
                x0, y0, z0,
                px[j], py[j], pz[j], pmass[j],
                _mm256_set1_ps( softeningSquared ) );

        }

        force[0][i] = horizontal_sum( ax );
        force[1][i] = horizontal_sum( ay );
        force[2][i] = horizontal_sum( az );
    }

    end = libtime_cpu();

    return libtime_cpu_to_wall(end - start) * 1e-6f;
}
#endif

/* vim: set ts=4 sts=4 sw=4 et: */
