#!/bin/bash
# usage: tools/orfs_run_baseline.sh <design> [cores]; ORFS asap7 full flow into orfs_runs/<design>_base
d=$1; c=${2:-5}
start=$(date +%s)
docker run --rm -u $(id -u):$(id -g) -v $HOME/attar/orfs_runs:/work openroad/orfs:26Q4-368-g9ca2a9e76 bash -c \
  "cd /OpenROAD-flow-scripts/flow && make NUM_CORES=$c DESIGN_CONFIG=./designs/asap7/$d/config.mk WORK_HOME=/work/${d}_base > /work/${d}_base.make.log 2>&1"
rc=$?
echo "$d rc=$rc $(( $(date +%s) - start ))s"
