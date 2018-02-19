#!/bin/bash

set -e

CC=$1

try_flags() {
	OMP_FLAG="$1"
	LIB_FLAG="$2"
	if ${CC} ${OMP_FLAG} -o openmp-test openmp-test.c ${LIB_FLAG} &>/dev/null; then
		if env KMP_AFFINITY=verbose OMP_NUM_THREADS=2 ./openmp-test &>/dev/null; then
			cat > openmp.mk.tmp <<-EOF
			OPENMP_SUPPORTED := Yes
			OPENMP_CFLAGS    := ${OMP_FLAG}
			OPENMP_LIBS      := ${LIB_FLAG}
			EOF
			cmp --quiet openmp.mk openmp.mk.tmp || mv openmp.mk.tmp openmp.mk
			exit 0
		fi
	fi
}

failed() {
	cat > openmp.mk.tmp <<-EOF
	OPENMP_SUPPORTED := No
	OPENMP_CFLAGS    :=
	OPENMP_LIBS      :=
	EOF
	cmp --quiet openmp.mk openmp.mk.tmp || mv openmp.mk.tmp openmp.mk
}

rm -f openmp.mk
echo "    * detecting OpenMP compiler flags"
try_flags -fopenmp=libomp -lomp
try_flags -fopenmp=libiomp5 -liomp5
try_flags -fopenmp=libgomp
try_flags -fopenmp -lomp
try_flags -fopenmp -liomp5
try_flags -fopenmp
failed
exit 0
