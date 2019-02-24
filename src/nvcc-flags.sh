#!/bin/bash

set -e

NVCC="$1"

verlte() {
	[[ "$1" = "$(echo -e "$1\n$2" | sort -V | head -n1)" ]]
}
verlt() {
    [ "$1" = "$2" ] && return 1 || verlte $1 $2
}

vergte() {
	[[ "$1" = "$(echo -e "$1\n$2" | sort -V | tail -n1)" ]]
}

NVCC_VER=$("$NVCC" --version | awk '/^Cuda comp/ { print $5 }' | cut -d',' -f 1)

NVCC_FLAGS=()

if [ -z "$NVCC_VER" ] || verlt ${NVCC_VER} 6.0; then
	echo "Your installed CUDA version is too old. Please run CUDA 6.0 or higher!" >&2
	exit 1
fi

NVCC_FLAGS+=(
	-gencode=arch=compute_50,code=sm_50
	-gencode=arch=compute_52,code=sm_52
)

if vergte ${NVCC_VER} 8.0; then
	NVCC_FLAGS+=( -gencode=arch=compute_61,code=sm_61 )
fi

if vergte ${NVCC_VER} 9.0; then
	NVCC_FLAGS+=( -gencode=arch=compute_70,code=sm_70 )
fi

if vergte ${NVCC_VER} 10.0; then
	NVCC_FLAGS+=( -gencode=arch=compute_75,code=sm_75 )
fi

cat > nvcc-flags.mk << EOF
NVCC_TARGET_FLAGS_DONE := 1
NVCC_TARGET_FLAGS := ${NVCC_FLAGS[@]}
EOF
exit 0
