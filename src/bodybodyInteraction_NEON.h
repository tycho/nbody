/*
 *
 * bodybodyInteraction_NEON.h
 *
 * ARM NEON implementation of N-body computation.
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

#if defined(__ARM_NEON__) || defined(__aarch64__)

#include <arm_neon.h>

static inline float32x4_t
vrsqrtq_f32(const float32x4_t val)
{
    float32x4_t e = vrsqrteq_f32(val);
    e = vmulq_f32(vrsqrtsq_f32(vmulq_f32(e, e), val), e);
    return e;
}

static inline float
vhaddq_f32(const float32x4_t x)
{
#if defined(__aarch64__)
    return vaddvq_f32(x);
#else
    static const float32x2_t f0 = vdup_n_f32(0.0f);
    return vget_lane_f32(vpadd_f32(f0, vget_high_f32(x) + vget_low_f32(x)), 1);
#endif
}

static inline void
bodyBodyInteraction(
    float32x4_t *fx,
    float32x4_t *fy,
    float32x4_t *fz,

    const float32x4_t x0,
    const float32x4_t y0,
    const float32x4_t z0,

    const float32x4_t x1,
    const float32x4_t y1,
    const float32x4_t z1,
    const float32x4_t mass1,

    const float32x4_t softeningSquared )
{
    // r_01  [3 FLOPS]
    const float32x4_t dx = vsubq_f32(x1, x0);
    const float32x4_t dy = vsubq_f32(y1, y0);
    const float32x4_t dz = vsubq_f32(z1, z0);

    // d^2 + e^2 [6 FLOPS]
    const float32x4_t distSq =
        vaddq_f32(
            vaddq_f32(
                vaddq_f32(
                    vmulq_f32(dx, dx),
                    vmulq_f32(dy, dy)
                ),
                vmulq_f32(dz, dz)
            ),
            softeningSquared
        );

    // invDistCube =1/distSqr^(3/2)  [4 FLOPS (2 mul, 1 sqrt, 1 inv)]
    const float32x4_t invDist = vrsqrtq_f32(distSq);
    const float32x4_t invDistCube =
        vmulq_f32(
            invDist,
            vmulq_f32(
                invDist,
                invDist
            )
        );

    // s = m_j * invDistCube [1 FLOP]
    const float32x4_t s = vmulq_f32(mass1, invDistCube);

    // (m_1 * r_01) / (d^2 + e^2)^(3/2)  [6 FLOPS]
    *fx = vaddq_f32(*fx, vmulq_f32(dx, s));
    *fy = vaddq_f32(*fy, vmulq_f32(dy, s));
    *fz = vaddq_f32(*fz, vmulq_f32(dz, s));
}

#endif

/* vim: set ts=4 sts=4 sw=4 et: */
