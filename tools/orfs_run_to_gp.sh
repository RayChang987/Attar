#!/bin/bash
# usage: tools/orfs_run_to_gp.sh <design> [cores]; ORFS asap7 up to 3_3_place_gp into orfs_runs/<design>_base
d=$1; c=${2:-5}
start=$(date +%s)
docker run --rm -u $(id -u):$(id -g) -v $HOME/attar/orfs_runs:/work openroad/orfs:26Q4-368-g9ca2a9e76 bash -c \
  "cd /OpenROAD-flow-scripts/flow && make NUM_CORES=$c DESIGN_CONFIG=./designs/asap7/$d/config.mk WORK_HOME=/work/${d}_base 3_3_place_gp > /work/${d}_base.gp.log 2>&1; \
   for v in TECH_LEF SC_LEF ADDITIONAL_LEFS LIB_FILES ADDITIONAL_LIBS; do make --no-print-directory DESIGN_CONFIG=./designs/asap7/$d/config.mk WORK_HOME=/work/${d}_base print-\$v 2>/dev/null | tail -1; done > /work/${d}_base.vars"
echo "$d rc=$? $(( $(date +%s) - start ))s"
