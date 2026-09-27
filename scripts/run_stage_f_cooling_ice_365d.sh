#!/bin/bash
set -euo pipefail
cd /mnt/c/Users/zhen.luo/ocean_solver

# Pre-registered Stage-F cooling_ice candidate. Do not run until the annual
# MOM6 v12 final-90d 3D/MLD comparison has been scored.
GATE=research/experiments/industrial_comparison_045/mom6_stage_f_dynamic_365d_v12_gate.json
if [ ! -f "$GATE" ]; then
  echo "Blocking gate not scored yet: $GATE" >&2
  exit 1
fi
/root/jax-gpu/bin/python src/run_long_integration_global.py \
  --mode-split --use-scan --dtype float32 --lat-max 65 --resolution 0.5 \
  --resolution-remap area --seasonal-wind --wind-year 2023 \
  --real-air-temp-monthly --no-meridional-heat-flux --lambda-bulk 80 \
  --kappa-v 1e-6 --kappa-conv 0.01 --kappa-gm 0 --localize-conv --fct-adv \
  --project-adv-vel --min-depth 500 --smooth-passes 80 --nu-h 2e6 --dt 1800 \
  --dynamic-ice --dynamic-ice-lat-band 40 65 --mixed-layer-depth 100 \
  --mixed-layer-lat-band 40 60 --mixed-layer-mode constant \
  --mixed-layer-gate-mode cooling_ice --days 365 --snap-days 30 --save-3d \
  --tag global_stage_f_dyn_ice_band40_65_mld100_lat40_60_cooling_ice_gate_365d \
  --out-dir /root/external_models/results/industrial_comparison_045 \
  --log-dir /root/external_models/logs/industrial_comparison_045
