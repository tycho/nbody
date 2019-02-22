/*
 *
 * nbody_CPU_NEON.cpp
 *
 * Multithreaded NEON CPU implementation of the O(N^2) N-body calculation.
 * Uses SOA (structure of arrays) representation because it is a much
 * better fit for NEON.
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

#ifdef __ARM_NEON__

#include "libtime.h"

#include "nbody_util.h"

#include "bodybodyInteraction_NEON.h"
#include "nbody_CPU_SIMD.h"

const char *SIMD_ALGORITHM_NAME = "NEON";

DEFINE_SOA(ComputeGravitation_SIMD)
{
    uint64_t start, end;

    start = libtime_cpu();

    ASSERT_ALIGNED(mass, NBODY_ALIGNMENT);
    for ( size_t i = 0; i < 3; i++ ) {
        ASSERT_ALIGNED(pos[i], NBODY_ALIGNMENT);
        ASSERT_ALIGNED(force[i], NBODY_ALIGNMENT);
    }

    ASSUME(N >= 1024);
    ASSUME(N % 1024 == 0);

    #pragma omp parallel for schedule(guided, 16)
    for (size_t i = 0; i < N; i++)
    {
        const vf32x4_t x0 = _vec_set_ps1( pos[0][i] );
        const vf32x4_t y0 = _vec_set_ps1( pos[1][i] );
        const vf32x4_t z0 = _vec_set_ps1( pos[2][i] );

        vf32x4_t ax = vec_zero;
        vf32x4_t ay = vec_zero;
        vf32x4_t az = vec_zero;

        for ( size_t j = 0; j < N; j += 4 )
        {
            const vf32x4_t x1 = *(vf32x4_t *)&pos[0][j];
            const vf32x4_t y1 = *(vf32x4_t *)&pos[1][j];
            const vf32x4_t z1 = *(vf32x4_t *)&pos[2][j];
            const vf32x4_t mass1 = *(vf32x4_t *)&mass[j];

            bodyBodyInteraction(
                &ax, &ay, &az,
                x0, y0, z0,
                x1, y1, z1, mass1,
                _vec_set_ps1( softeningSquared ) );

        }

        // Accumulate sum of four floats in the NEON register
        force[0][i] = _vec_sum( ax );
        force[1][i] = _vec_sum( ay );
        force[2][i] = _vec_sum( az );
    }

    end = libtime_cpu();

    return libtime_cpu_to_wall(end - start) * 1e-6f;
}
#endif

/* vim: set ts=4 sts=4 sw=4 et: */
