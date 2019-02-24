#!/bin/bash
#
# This is a distribution-agnostic build script. Do not use "apt-get", "dnf", or
# similar in here. Add any package installation gunk into the appropriate
# install script instead.
#
set -ex

#
# "BROKEN" can mean multiple things, but in all cases the issues stem from an
# outdated build environment. Either meson is too old, or "clang -fopenmp" is
# broken, or something similar.
#
if [[ ! -z "$BROKEN" ]] && [[ "$CC" == "clang" ]]; then
	echo "Clang is known to be broken in this build environment, aborting now." >&2
	exit 0
fi

HAVE_MESON=0
if [[ -z "$BROKEN" ]]; then
	if type -P meson &>/dev/null; then
		HAVE_MESON=1
	fi
else
	echo "Meson is known to be broken in this build environment, will skip Meson builds." >&2
fi

make distclean
make CC=${CC} CXX=${CXX} V=1
./nbody --bodies 8 --cycle-after 3 --iterations 1 --verbose

make distclean
make CC=${CC} CXX=${CXX} NO_OPENMP=1 V=1
./nbody --bodies 8 --cycle-after 3 --iterations 1 --verbose

if type -P nvcc &>/dev/null; then
	make distclean
	make CC=${CC} CXX=${CXX} CUDA=1 V=1
	./nbody --bodies 8 --cycle-after 3 --iterations 1 --verbose
fi

if [[ $HAVE_MESON -eq 1 ]]; then
	rm -rf build
	mkdir build
	meson . build
	ninja -C build
	build/nbody --bodies 8 --cycle-after 3 --iterations 1 --verbose

	if type -P nvcc &>/dev/null; then
		rm -rf build
		mkdir build
		meson . build -Dcuda=true
		ninja -C build
		build/nbody --bodies 8 --cycle-after 3 --iterations 1 --verbose
	fi
fi
