#include <omp.h>
#include <math.h>
#include <stdio.h>

#ifndef _OPENMP
#error "compile with -fopenmp please!"
#endif

#ifdef __GNUC__
#define USED __attribute__((used))
#else
#define USED
#endif

/* Test to ensure that schedule(guided) doesn't result in undefined symbols
 * (libgomp currently defines some symbols that libomp does not)
 */
static void USED saxpy_guided(int n, float a, float * __restrict x, float * __restrict y)
{
	#pragma omp parallel for schedule(guided)
	for (int i = 0; i < n; ++i) {
		y[i] = a * x[i] + y[i];
	}
}

/* Test to ensure math.h functions don't cause compile errors with -fopenmp
 * (old Clang + -fopenmp + newer GCC = breakage)
 */
static float USED rsqrt(float x)
{
	return 1.0f / sqrtf(x);
}


static inline int
threadCount(void)
{
    int k;
#  pragma omp parallel
    {
#  pragma omp master
        {
            k = omp_get_num_threads();
        }
    }
    return k;
}

int main(int argc, char **argv)
{
	printf("thread count: %d\n", threadCount());

	return 0;
}
