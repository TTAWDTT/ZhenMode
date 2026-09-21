#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/zhen.luo/ocean_solver
export OCEAN_SOLVER_BATHYMETRY=/mnt/c/Users/zhen.luo/Desktop/ETOPO_2022_v1_r3600x1800_surface.nc
mkdir -p results/north_boundary_ice_proxy logs/north_boundary_ice_proxy

run() {
  local tag="$1"; shift
  echo "=== RUN ${tag} ==="
  .venv-gpu-jax/bin/python src/run_long_integration_global.py \
    --mode-split --use-scan --dtype float32 \
    --kappa-gm 1000 --project-adv-vel --fct-adv \
    --seasonal-wind --wind-year 2023 --real-air-temp \
    --kappa-v 1e-6 --kappa-conv 0.01 \
    --tag "${tag}" \
    --out-dir results/north_boundary_ice_proxy \
    --log-dir logs/north_boundary_ice_proxy \
    "$@"
}

run real_air_kv1e-6_kconv001_lat65 --days 365 --dt 3600 --lat-max 65 --ny 130
run real_air_kv1e-6_kconv001_polarcap4_6 --days 365 --dt 3600 --polar-cap-rows 4 --polar-cap-taper 6
run real_air_kv1e-6_kconv001_nocap90d --days 90 --dt 3600 --polar-cap-rows 0 --polar-cap-taper 0
