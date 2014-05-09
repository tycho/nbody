#include <omp.h>
#include <stdio.h>

#ifndef _OPENMP
#error "compile with -fopenmp please!"
#endif

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
