#!/bin/bash

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $SCRIPT_DIR

CC="$1"
OMP_FLAG="$2"
LIB_FLAG="$3"

if [[ -z "$LIB_FLAG" ]]; then
	# Not specified, so let the compiler choose the appropriate defaults
	LIB_FLAG="$OMP_FLAG"
fi

# Compile and link with separate steps, ensuring that -fopenmp doesn't get
# passed to the linker if we have a $LIB_FLAG specified.
if ${CC} ${OMP_FLAG} -c -o openmp-test.o openmp-test.c &>/dev/null; then
	if ${CC} -o openmp-test openmp-test.o ${LIB_FLAG} -lm &>/dev/null; then
		if env KMP_AFFINITY=verbose OMP_NUM_THREADS=2 ./openmp-test &>/dev/null; then
			exit 0
		fi
	fi
fi

exit 1
