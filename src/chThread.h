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

#pragma once

#include <assert.h>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <mutex>
#include <condition_variable>

class semaphore
{
private:
	std::mutex mtx;
	std::condition_variable cnd;
	uint32_t count;

public:
	void post()
	{
		std::lock_guard<decltype(mtx)> lock(mtx);
		++count;
		cnd.notify_one();
	}

	void wait()
	{
		std::unique_lock<decltype(mtx)> lock(mtx);
		while (!count)
			cnd.wait(lock);
		--count;
	}
};

typedef int (*thread_proc_t)(void *);

class worker_thread {
private:
	std::thread thread;
	semaphore semWait;
    semaphore semDone;

	thread_proc_t task;
	void *param;
	bool stopped;

	int _routine()
	{
		int loop = 1;
		do {
			semWait.wait();
			if (!param)
				loop = 0;
			else
				(*task)(param);
			semDone.post();
		} while (loop);
		return 0;
	}

public:
	worker_thread()
		: task(nullptr),
		  param(nullptr)
	{
		thread = std::thread(&worker_thread::_routine, this);
		stopped = false;
	}

	~worker_thread()
	{
		if (!stopped)
			stop();
	}

	void stop()
	{
		stopped = true;
		delegate(nullptr, nullptr, true);
		thread.join();
	}

	void delegate(thread_proc_t _proc, void *_param, bool _sync)
	{
		task = _proc;
		param = _param;
		semWait.post();
		if (_sync)
			semDone.wait();
	}
};
