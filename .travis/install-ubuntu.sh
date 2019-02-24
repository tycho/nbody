#!/bin/bash
#
# This is an install script for Ubuntu-specific packages.
#
set -ex
apt-get update
apt-get install -y locales
locale-gen en_US.UTF-8

PACKAGES=(build-essential pkg-config clang libomp-dev)

apt-get install -y "${PACKAGES[@]}"

# Optional: try to install meson. Doesn't exist on older Ubuntu, so this could
# be made mandatory in the future.
apt-get install -y meson || true
