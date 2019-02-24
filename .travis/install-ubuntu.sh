#!/bin/bash
#
# This is an install script for Ubuntu-specific packages.
#
set -ex

if [[ ! -z "$BROKEN" ]] && [[ "$CC" == "clang" ]]; then
	echo "====================================================================" >&2
	echo "Clang is known to be broken in this build environment, aborting now." >&2
	echo "====================================================================" >&2
	exit 0
fi

apt-get update
apt-get install -y locales
locale-gen en_US.UTF-8

PACKAGES=(build-essential pkg-config clang)

apt-get install -y "${PACKAGES[@]}"

# Optional: try to install libomp-dev. It doesn't exist for the CUDA 7.5 docker
# environment (which is based on Trusty).
apt-get install -y libomp-dev || true

# Optional: try to install meson. Doesn't exist on older Ubuntu, so this could
# be made mandatory in the future.
if [[ -z "$BROKEN" ]]; then
	apt-get install -y meson || true
fi
