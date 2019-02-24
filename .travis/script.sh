#!/bin/bash
#
# This is a distribution-agnostic build script. Do not use "apt-get", "dnf", or
# similar in here. Add any package installation gunk into the appropriate
# install script instead.
#
set -ex

make distclean
make CC=${CC} CXX=${CXX} V=1
./nbody --bodies 8 --cycle-after 3 --iterations 1 --verbose

make distclean
make CC=${CC} CXX=${CXX} NO_OPENMP=1 V=1
./nbody --bodies 8 --cycle-after 3 --iterations 1 --verbose

if which nvcc; then
	make distclean
	make CC=${CC} CXX=${CXX} CUDA=1 V=1
	./nbody --bodies 8 --cycle-after 3 --iterations 1 --verbose
fi
