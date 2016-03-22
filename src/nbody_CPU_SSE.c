/*
 *
 * nbody_CPU_SSE.cpp
 *
 * Multithreaded SSE CPU implementation of the O(N^2) N-body calculation.
 * Uses SOA (structure of arrays) representation because it is a much
 * better fit for SSE.
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

#if defined(HAVE_SSE)
#include "libtime.h"

#include "nbody_util.h"

#include "bodybodyInteraction_SSE.h"
#include "nbody_CPU_SIMD.h"

float
ComputeGravitation_SIMD(
    afloat ** restrict force,
    afloat ** restrict pos,
    afloat *  restrict mass,
    float softeningSquared,
    size_t N
)
{
    uint64_t start, end;

    start = libtime_cpu();

    ASSUME(N >= 1024);
    ASSUME(N % 1024 == 0);

    ASSERT_ALIGNED(mass, NBODY_ALIGNMENT);
    for ( size_t i = 0; i < 3; i++ ) {
        ASSERT_ALIGNED(pos[i], NBODY_ALIGNMENT);
        ASSERT_ALIGNED(force[i], NBODY_ALIGNMENT);
    }

    #pragma omp parallel for schedule(guided, 16)
    #pragma vector aligned
    #pragma ivdep
    for ( size_t i = 0; i < N; i++ )
    {
        const __m128 x0 = _mm_set_ps1( pos[0][i] );
        const __m128 y0 = _mm_set_ps1( pos[1][i] );
        const __m128 z0 = _mm_set_ps1( pos[2][i] );

        __m128 ax = _mm_setzero_ps();
        __m128 ay = _mm_setzero_ps();
        __m128 az = _mm_setzero_ps();

        #pragma vector aligned
        #pragma ivdep
        for ( size_t j = 0; j < N; j += 4 )
        {
            const __m128 x1 = *(__m128 *)&pos[0][j];
            const __m128 y1 = *(__m128 *)&pos[1][j];
            const __m128 z1 = *(__m128 *)&pos[2][j];
            const __m128 mass1 = *(__m128 *)&mass[j];

            bodyBodyInteraction(
                &ax, &ay, &az,
                x0, y0, z0,
                x1, y1, z1, mass1,
                _mm_set_ps1( softeningSquared ) );

        }
        // Accumulate sum of four floats in the SSE register
        ax = horizontal_sum_ps( ax );
        ay = horizontal_sum_ps( ay );
        az = horizontal_sum_ps( az );

        _mm_store_ss( (float *) &force[0][i], ax );
        _mm_store_ss( (float *) &force[1][i], ay );
        _mm_store_ss( (float *) &force[2][i], az );
    }

    end = libtime_cpu();

    return libtime_cpu_to_wall(end - start) * 1e-6f;
}
#endif

/* vim: set ts=4 sts=4 sw=4 et: */
