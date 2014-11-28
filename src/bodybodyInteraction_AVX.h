/*
 *
 * bodybodyInteraction_AVX.h
 *
 * Intel x86/x86_64 AVX implementation of N-body computation.
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

#include <immintrin.h>

static inline __m256
rcp_sqrt_nr_ps(const __m256 x)
{
    const __m256
        nr      = _mm256_rsqrt_ps(x),
        muls    = _mm256_mul_ps(_mm256_mul_ps(nr, nr), x),
        beta    = _mm256_mul_ps(_mm256_set1_ps(0.5f), nr),
        gamma   = _mm256_sub_ps(_mm256_set1_ps(3.0f), muls);

    return _mm256_mul_ps(beta, gamma);
}

static inline float
horizontal_sum( __m256 x )
{
    // hiQuad = ( x7, x6, x5, x4 )
    const __m128 hiQuad = _mm256_extractf128_ps(x, 1);
    // loQuad = ( x3, x2, x1, x0 )
    const __m128 loQuad = _mm256_castps256_ps128(x);
    // sumQuad = ( x3 + x7, x2 + x6, x1 + x5, x0 + x4 )
    const __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
    // loDual = ( -, -, x1 + x5, x0 + x4 )
    const __m128 loDual = sumQuad;
    // hiDual = ( -, -, x3 + x7, x2 + x6 )
    const __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);
    // sumDual = ( -, -, x1 + x3 + x5 + x7, x0 + x2 + x4 + x6 )
    const __m128 sumDual = _mm_add_ps(loDual, hiDual);
    // lo = ( -, -, -, x0 + x2 + x4 + x6 )
    const __m128 lo = sumDual;
    // hi = ( -, -, -, x1 + x3 + x5 + x7 )
    const __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
    // sum = ( -, -, -, x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 )
    const __m128 sum = _mm_add_ss(lo, hi);
    return _mm_cvtss_f32(sum);
}

static inline void
bodyBodyInteraction(
    __m256 *fx,
    __m256 *fy,
    __m256 *fz,

    __m256 x0,
    __m256 y0,
    __m256 z0,

    __m256 x1,
    __m256 y1,
    __m256 z1,
    __m256 mass1,

    __m256 softeningSquared )
{
    // r_01  [3 FLOPS]
    __m256 dx = _mm256_sub_ps( x1, x0 );
    __m256 dy = _mm256_sub_ps( y1, y0 );
    __m256 dz = _mm256_sub_ps( z1, z0 );

    // d^2 + e^2 [6 FLOPS]
    __m256 distSq =
        _mm256_add_ps(
            _mm256_add_ps(
                _mm256_mul_ps( dx, dx ),
                _mm256_mul_ps( dy, dy )
            ),
            _mm256_mul_ps( dz, dz )
        );
    distSq = _mm256_add_ps( distSq, softeningSquared );

    // invDistCube =1/distSqr^(3/2)  [4 FLOPS (2 mul, 1 sqrt, 1 inv)]
    __m256 invDist = rcp_sqrt_nr_ps( distSq );
    __m256 invDistCube =
        _mm256_mul_ps(
            invDist,
            _mm256_mul_ps(
                invDist, invDist )
        );

    // s = m_j * invDistCube [1 FLOP]
    __m256 s = _mm256_mul_ps( mass1, invDistCube );

    // (m_1 * r_01) / (d^2 + e^2)^(3/2)  [6 FLOPS]
    *fx = _mm256_add_ps( *fx, _mm256_mul_ps( dx, s ) );
    *fy = _mm256_add_ps( *fy, _mm256_mul_ps( dy, s ) );
    *fz = _mm256_add_ps( *fz, _mm256_mul_ps( dz, s ) );
}

/* vim: set ts=4 sts=4 sw=4 et: */
