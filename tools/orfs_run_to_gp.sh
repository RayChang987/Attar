#!/bin/bash
# usage: tools/orfs_run_to_gp.sh <design | platform/design> [cores]
# ORFS up to 3_3_place_gp. asap7 (no platform given): orfs_runs/<design>_base; another platform (e.g. nangate45/aes):
# orfs_runs/<platform>/<design>_base. The design's config: orfs_runs/designs/<design>/config.mk for a design of our own,
# else ORFS's designs/<platform>/<design>/config.mk.
key=$1; c=${2:-5}
case $key in */*) p=${key%%/*}; d=${key#*/}; run=$p/${d}_base;; *) p=asap7; d=$key; run=${d}_base;; esac
cfg=./designs/$p/$d/config.mk
[ $p = asap7 ] && [ -f $HOME/attar/orfs_runs/designs/$d/config.mk ] && cfg=/work/designs/$d/config.mk   # a design of our own
mkdir -p $HOME/attar/orfs_runs/$(dirname $run)
start=$(date +%s)
docker run --rm -u $(id -u):$(id -g) -v $HOME/attar/orfs_runs:/work openroad/orfs:26Q4-368-g9ca2a9e76 bash -c \
  "cd /OpenROAD-flow-scripts/flow && make NUM_CORES=$c DESIGN_CONFIG=$cfg WORK_HOME=/work/$run 3_3_place_gp > /work/$run.gp.log 2>&1; \
   for v in TECH_LEF SC_LEF ADDITIONAL_LEFS LIB_FILES ADDITIONAL_LIBS; do make --no-print-directory DESIGN_CONFIG=$cfg WORK_HOME=/work/$run print-\$v 2>/dev/null | tail -1; done > /work/$run.vars"
rc=$?
ls $HOME/attar/orfs_runs/$run/results/*/*/base/3_3_place_gp.odb > /dev/null 2>&1 && ok=ok || ok=FAILED
echo "$key $ok $(( $(date +%s) - start ))s"
