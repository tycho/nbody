#!/bin/bash
#
# This is a distribution-agnostic build script. Do not use "apt-get", "dnf", or
# similar in here. Add any package installation gunk into the appropriate
# install script instead.
#
set -ex

rm -rf build
mkdir build
meson . build
ninja -C build
build/nbody --bodies 8 --cycle-after 3 --iterations 1 --verbose

if type -P nvcc &>/dev/null; then
	rm -rf build
	mkdir build
	meson . build -Duse_cuda=true
	ninja -C build
	build/nbody --bodies 8 --cycle-after 3 --iterations 1 --verbose
fi
