#!/bin/bash

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $SCRIPT_DIR

CXX=$1

try_flags() {
	OMP_FLAG="$1"
	LIB_FLAG="$2"

	if [[ -z "$LIB_FLAG" ]]; then
		# Not specified, so let the compiler choose the appropriate defaults
		LIB_FLAG="$OMP_FLAG"
	fi

	# Compile and link with separate steps, ensuring that -fopenmp doesn't get
	# passed to the linker if we have a $LIB_FLAG specified.
	if ${CXX} ${OMP_FLAG} -c -o openmp-test.o openmp-test.cpp &>/dev/null; then
		if ${CXX} -o openmp-test openmp-test.o ${LIB_FLAG} -lm &>/dev/null; then
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
