/*
 *
 * chError.h
 *
 * Error handling for CUDA:
 *     CUDA_CHECK() and CUDART_CHECK() macros implement
 *         goto-based error handling, and
 *     chGetErrorString() maps a driver API error to a string.
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


#ifndef __CHERROR_H__
#define __CHERROR_H__

#ifdef DEBUG
#include <stdio.h>
#endif

#include "chCUDA.h"

#ifndef NO_CUDA

static inline const char *
chGetErrorString(cudaError_t status)
{
    return cudaGetErrorString(status);
}

//
// To use these macros, a local cudaError_t or CUresult called 'status'
// and a label Error: must be defined.  In the debug build, the code will
// emit an error to stderr.  In both debug and retail builds, the code will
// goto Error if there is an error.
//

#ifdef DEBUG
#define CUDART_CHECK( fn ) do { \
        (status) =  (fn); \
        if ( cudaSuccess != (status) ) { \
            fprintf( stderr, "CUDA Runtime Failure (line %d of file %s):\n\t" \
                "%s returned 0x%x (%s)\n", \
                __LINE__, __FILE__, #fn, status, chGetErrorString(status) ); \
            goto Error; \
        } \
    } while (0);

#define CUDA_CHECK( fn ) do { \
        (status) =  (fn); \
        if ( CUDA_SUCCESS != (status) ) { \
            fprintf( stderr, "CUDA Runtime Failure (line %d of file %s):\n\t%s "\
                "returned 0x%x (%s)\n", \
                __LINE__, __FILE__, #fn, status, chGetErrorString(status) ); \
            goto Error; \
        } \
    } while (0);

#else

#define CUDART_CHECK( fn ) do { \
    status = (fn); \
    if ( cudaSuccess != (status) ) { \
            goto Error; \
        } \
    } while (0);

#define CUDA_CHECK( fn ) do { \
        (status) =  (fn); \
        if ( CUDA_SUCCESS != (status) ) { \
            goto Error; \
        } \
    } while (0);

#endif

#else

static inline const char *
chGetErrorString( cudaError_t status )
{
    return "CUDA support is not built in.";
}

static inline const char* cudaGetErrorString( cudaError_t error )
{
    return "CUDA support is not built in.";
}

#define CUDART_CHECK( fn ) do { \
    status = (fn); \
    if ( cudaSuccess != (status) ) { \
            goto Error; \
        } \
    } while (0);

#define CUDA_CHECK( fn ) do { \
        (status) =  (fn); \
        if ( CUDA_SUCCESS != (status) ) { \
            goto Error; \
        } \
    } while (0);

#endif

#endif
