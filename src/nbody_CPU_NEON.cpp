/*
 *
 * nbody_CPU_NEON.cpp
 *
 * Multithreaded NEON CPU implementation of the O(N^2) N-body calculation.
 * Uses SOA (structure of arrays) representation because it is a much
 * better fit for NEON.
 *
 * Copyright (c) 2011-2012, Archaea Software, LLC.
 * Copyright (c) 2012-2021, Uplink Laboratories, LLC.
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

#if defined(__ARM_NEON__) || defined(__aarch64__)

#include <chrono>

#include "nbody_util.h"

#include "bodybodyInteraction_NEON.h"
#include "nbody_CPU_SIMD.h"

using namespace std;

const char *SIMD_ALGORITHM_NAME = "NEON";

DEFINE_SOA(ComputeGravitation_SIMD)
{
    auto start = chrono::steady_clock::now();

    ASSERT_ALIGNED(mass, NBODY_ALIGNMENT);
    ASSERT_ALIGNED(pos[0], NBODY_ALIGNMENT);
    ASSERT_ALIGNED(pos[1], NBODY_ALIGNMENT);
    ASSERT_ALIGNED(pos[2], NBODY_ALIGNMENT);
    ASSERT_ALIGNED(force[0], NBODY_ALIGNMENT);
    ASSERT_ALIGNED(force[1], NBODY_ALIGNMENT);
    ASSERT_ALIGNED(force[2], NBODY_ALIGNMENT);

    ASSUME(N % 1024 == 0);
    ASSUME(N >= 1024);

    #pragma omp parallel for schedule(guided)
    for (size_t i = 0; i < N; i++)
    {
        const float32x4_t x0 = vld1q_dup_f32(&pos[0][i]);
        const float32x4_t y0 = vld1q_dup_f32(&pos[1][i]);
        const float32x4_t z0 = vld1q_dup_f32(&pos[2][i]);

        float32x4_t ax = vmovq_n_f32(0.0f);
        float32x4_t ay = vmovq_n_f32(0.0f);
        float32x4_t az = vmovq_n_f32(0.0f);

        for ( size_t j = 0; j < N; j += 4 )
        {
            const float32x4_t x1 = vld1q_f32(&pos[0][j]);
            const float32x4_t y1 = vld1q_f32(&pos[1][j]);
            const float32x4_t z1 = vld1q_f32(&pos[2][j]);
            const float32x4_t mass1 = vld1q_f32(&mass[j]);

            bodyBodyInteraction(
                &ax, &ay, &az,
                x0, y0, z0,
                x1, y1, z1, mass1,
                vld1q_dup_f32(&softeningSquared));
        }

        // Accumulate sum of four floats in the NEON register
        force[0][i] = vhaddq_f32( ax );
        force[1][i] = vhaddq_f32( ay );
        force[2][i] = vhaddq_f32( az );
    }

    auto end = chrono::steady_clock::now();
    return chrono::duration<float, std::milli>(end - start).count();
}
#endif

/* vim: set ts=4 sts=4 sw=4 et: */
