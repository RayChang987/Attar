#!/bin/bash
# usage: tools/orfs_run_cts.sh <design> [cores] [platform]
# Copies the ORFS run orfs_runs/<design>_base (stopped after 3_3 global placement) to orfs_runs/cts/<design>_base and
# continues the flow there through CTS (make cts: 3_4 resize, 3_5 detailed placement, 4_1 CTS with TritonCTS).
# The copy leaves the 3_3 run, the validators' truth, untouched. Validation input for propagated clocks
# (tools/validate_cts.py).
d=$1; c=${2:-8}; p=${3:-asap7}
root=$HOME/attar/orfs_runs
src=$root/${d}_base; [ "$p" != asap7 ] && src=$root/$p/${d}_base
dst=$root/cts/${p}_${d}_base
mkdir -p $root/cts
if [ ! -d $dst ]; then cp -a $src $dst; fi
start=$(date +%s)
docker run --rm -u $(id -u):$(id -g) -v $root:/work openroad/orfs:26Q4-368-g9ca2a9e76 bash -c \
  "cd /OpenROAD-flow-scripts/flow && make NUM_CORES=$c DESIGN_CONFIG=./designs/$p/$d/config.mk WORK_HOME=/work/cts/${p}_${d}_base cts > /work/cts/${p}_${d}.make.log 2>&1"
rc=$?
echo "$p/$d rc=$rc $(( $(date +%s) - start ))s"
