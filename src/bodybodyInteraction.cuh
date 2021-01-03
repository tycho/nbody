/*
 *
 * bodybodyInteraction.cuh
 *
 * CUDA header for function to compute body-body interaction.
 * Also compatible with scalar (non-SIMD) CPU implementations.
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

#pragma once

static __host__ __device__ void bodyBodyInteraction(
    float *fx, float *fy, float *fz,
    float x0, float y0, float z0,
    float x1, float y1, float z1, float mass1,
    float softeningSquared)
{
    float dx = x1 - x0;
    float dy = y1 - y0;
    float dz = z1 - z0;

    float distSqr = dx*dx + dy*dy + dz*dz;
    distSqr += softeningSquared;

    //
    // rsqrtf() maps to SFU instruction - to support
    // double, this has to be changed.
    //
    float invDist = rsqrtf(distSqr);

    float invDistCube =  invDist * invDist * invDist;
    float s = mass1 * invDistCube;

    *fx = dx * s;
    *fy = dy * s;
    *fz = dz * s;
}

/* vim: set ts=4 sts=4 sw=4 et: */
