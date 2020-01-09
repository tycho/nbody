#!/bin/bash
#
# This is a distribution-agnostic build script. Do not use "apt-get", "dnf", or
# similar in here. Add any package installation gunk into the appropriate
# install script instead.
#
set -ex

BUILD_DIRS=()

cleanup() {
	rm -rf "${BUILD_DIRS[@]}"
}
trap cleanup EXIT

build_and_run() {
	ID="$1"
	MESON_ARGS="$2"
	BUILDDIR="travis-build-$ID"
	BUILD_DIRS+=( "$BUILDDIR" )
	rm -rf "$BUILDDIR"
	mkdir "$BUILDDIR"
	meson . "$BUILDDIR" $MESON_ARGS
	ninja -C "$BUILDDIR"
	"$BUILDDIR"/nbody --bodies 8 --cycle-after 3 --iterations 1 --verbose
}

build_and_run default
if type -P nvcc &>/dev/null; then
	build_and_run gpu "-Duse_cuda=true"
fi
build_and_run no-openmp "-Dopenmp=false"
build_and_run no-intrinsics "-Dintrinsics=false"
build_and_run opengl "-Dopengl=true"
