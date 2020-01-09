#!/bin/bash
#
# This is an install script for Ubuntu-specific packages.
#
set -ex

apt-get update
apt-get install -y locales
locale-gen en_US.UTF-8

PACKAGES=(build-essential pkg-config clang meson)

apt-get install -y "${PACKAGES[@]}"

# Optional: try to install libomp-dev. It doesn't exist for the CUDA 7.5 docker
# environment (which is based on Trusty).
apt-get install -y libomp-dev || true
