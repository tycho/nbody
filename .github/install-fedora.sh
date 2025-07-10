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
	cmake
	libomp-devel
	pkgconf-pkg-config
	glew-devel
	SDL2-devel
	glm-devel
)

dnf update -y
dnf install -y "${PACKAGES[@]}"
