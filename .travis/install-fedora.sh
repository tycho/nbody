#!/bin/bash
#
# This is an install script for Fedora-specific packages.
#
set -ex

# Base build packages
PACKAGES=(
	gcc-c++
	clang
	meson
	make
	libomp-devel
	pkgconf-pkg-config
)

dnf install -y "${PACKAGES[@]}"
