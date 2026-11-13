#!/usr/bin/env bash
# Build ggr_attar against gsta's objects (make -C $GSTA first). GSTA: the gsta checkout.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
GSTA=${GSTA:-$HOME/ISPD27-Contest/gsta}
CUDA=${CUDA:-/usr/local/cuda-11.8}
ARCH=${ARCH:-sm_86}
OBJ=$(ls $GSTA/build/*.o | grep -v '/main.o$')
$CUDA/bin/nvcc -O3 -std=c++17 -arch=$ARCH -I$GSTA/src -I$GSTA/third_party/flute -I$GSTA/experiments/rtree \
  -o $HERE/ggr_attar $HERE/ggr_attar.cpp $OBJ -lz -lpthread -lgomp
echo built $HERE/ggr_attar
