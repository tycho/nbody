/*
 *
 * nbody_CPU_SOA_tiled.c
 *
 * Tiled SOA implementation of the n-body algorithm.
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

#include "libtime.h"
#ifndef NO_CUDA
#define NO_CUDA
#endif
#include "chCUDA.h"

#include "bodybodyInteraction.cuh"
#include "nbody_CPU_SOA_tiled.h"

#define BODIES_PER_TILE 2048

float
ComputeGravitation_SOA_tiled(
    float ** restrict force,
    float ** restrict pos,
    float *  restrict mass,
    float softeningSquared,
    size_t N
)
{
    uint64_t start, end;

    if ( N % BODIES_PER_TILE != 0 )
        return 0.0f;

    start = libtime_cpu();

    #pragma omp parallel
    for (size_t tileStart = 0; tileStart < N; tileStart += BODIES_PER_TILE )
    {
        int tileEnd = tileStart + BODIES_PER_TILE;

        #pragma omp for schedule(guided)
        #pragma unroll_and_jam(4)
        for ( size_t i = 0; i < N; i++ )
        {
            float acx, acy, acz;
            const float myX = pos[0][i];
            const float myY = pos[1][i];
            const float myZ = pos[2][i];

            acx = acy = acz = 0;

            for ( size_t j = tileStart; j < tileEnd; j++ ) {

                const float bodyX = pos[0][j];
                const float bodyY = pos[1][j];
                const float bodyZ = pos[2][j];
                const float bodyMass = mass[j];

                float fx, fy, fz;

                bodyBodyInteraction(
                    &fx, &fy, &fz,
                    myX, myY, myZ,
                    bodyX, bodyY, bodyZ, bodyMass,
                    softeningSquared );

                acx += fx;
                acy += fy;
                acz += fz;
            }

            force[0][i] += acx;
            force[1][i] += acy;
            force[2][i] += acz;
        }
    }

    end = libtime_cpu();
    return libtime_cpu_to_wall(end - start) * 1e-6f;
}

/* vim: set ts=4 sts=4 sw=4 et: */
