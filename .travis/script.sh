#!/bin/bash
#
# This is a distribution-agnostic build script. Do not use "apt-get", "dnf", or
# similar in here. Add any package installation gunk into the appropriate
# install script instead.
#
set -ex

cleanup() {
	rm -rf travis-build-{cpu,gpu}
}
trap cleanup EXIT

BUILDDIR=travis-build-cpu
rm -rf $BUILDDIR
mkdir $BUILDDIR
meson . $BUILDDIR
ninja -C $BUILDDIR
$BUILDDIR/nbody --bodies 8 --cycle-after 3 --iterations 1 --verbose

if type -P nvcc &>/dev/null; then
	BUILDDIR=travis-build-gpu
	rm -rf $BUILDDIR
	mkdir $BUILDDIR
	meson . $BUILDDIR -Duse_cuda=true
	ninja -C $BUILDDIR
	$BUILDDIR/nbody --bodies 8 --cycle-after 3 --iterations 1 --verbose
fi
