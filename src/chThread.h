/*
 *
 * chThread.h
 *
 * Header file for helper classes and functions for threads.
 *
 * Copyright (c) 2011-2012, Archaea Software, LLC.
 * All rights reserved.

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

#ifndef __CHTHREAD_LINUX_H__
#define __CHTHREAD_LINUX_H__

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#ifdef USE_LIBC11
#include "c11/threads.h"
#else
#include <threads.h>
#endif

typedef struct _sem_t {
    mtx_t mtx;
    cnd_t cnd;
    uint32_t count;
} sem_t;

static int sem_init(sem_t *sem)
{
    if (!sem)
        return 1;
    if (mtx_init(&sem->mtx, mtx_plain | mtx_recursive) != thrd_success)
        return 1;
    if (cnd_init(&sem->cnd) != thrd_success)
        return 1;
    sem->count = 0;
    return 0;
}

static int sem_post(sem_t *sem)
{
    mtx_lock(&sem->mtx);
    sem->count++;
    cnd_signal(&sem->cnd);
    mtx_unlock(&sem->mtx);
    return 0;
}

static int sem_wait(sem_t *sem)
{
    mtx_lock(&sem->mtx);
    while (sem->count == 0)
        cnd_wait(&sem->cnd, &sem->mtx);
    sem->count--;
    mtx_unlock(&sem->mtx);
    return 0;
}

static void sem_destroy(sem_t *sem)
{
    mtx_destroy(&sem->mtx);
    cnd_destroy(&sem->cnd);
}

typedef struct _worker_thread_t {
    thrd_t thread;
    sem_t semWait;
    sem_t semDone;
    thrd_start_t delegate;
    void *param;
} worker_thread_t;

static int worker_delegate(worker_thread_t *worker, thrd_start_t delegate, void *param, int sync);

static int worker_create(worker_thread_t *worker)
{
    if (!worker)
        return 1;
    memset(worker, 0, sizeof(worker_thread_t));
    if (sem_init(&worker->semWait))
        return 1;
    if (sem_init(&worker->semDone))
        return 1;
    return 0;
}

static void worker_destroy(worker_thread_t *worker)
{
    assert(worker);
    worker_delegate(worker, NULL, NULL, 1);
    thrd_join(worker->thread, NULL);
    sem_destroy(&worker->semWait);
    sem_destroy(&worker->semDone);
}

static int _worker_routine(void *_p)
{
    worker_thread_t *w = (worker_thread_t *)_p;
    int loop = 1;
    do {
        sem_wait(&w->semWait);
        if (w->param == NULL)
            loop = 0;
        else
            (*w->delegate)(w->param);
        sem_post(&w->semDone);
    } while (loop);
    return 0;
}

static int worker_start(worker_thread_t *worker)
{
    assert(worker);
    if (thrd_create(&worker->thread, _worker_routine, worker) != thrd_success)
        return 1;
    return 0;
}

static int worker_join(worker_thread_t *worker)
{
    assert(worker);
    if (thrd_join(worker->thread, NULL))
        return 1;
    return 0;
}

static int worker_delegate(worker_thread_t *worker, thrd_start_t delegate, void *param, int sync)
{
    assert(worker);
    worker->delegate = delegate;
    worker->param = param;
    if (sem_post(&worker->semWait))
        return 1;
    if (sync && sem_wait(&worker->semDone))
        return 1;
    return 0;
}

#endif
