#!/bin/bash
set -euo pipefail
cd /mnt/c/Users/zhen.luo/ocean_solver

# Launch the pre-registered cooling_ice candidate only after the MOM6 annual
# comparison has produced its gate JSON. The inner script performs the same
# existence check as a second guard.
while [ ! -f research/experiments/industrial_comparison_045/mom6_stage_f_dynamic_365d_v12_gate.json ]; do
  sleep 300
done

bash scripts/run_stage_f_cooling_ice_365d.sh
