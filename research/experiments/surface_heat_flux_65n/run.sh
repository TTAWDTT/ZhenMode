#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/zhen.luo/ocean_solver
export OCEAN_SOLVER_BATHYMETRY=/mnt/c/Users/zhen.luo/Desktop/ETOPO_2022_v1_r3600x1800_surface.nc
mkdir -p results/surface_heat_flux_65n logs/surface_heat_flux_65n

run() {
  local tag="$1"; shift
  echo "=== RUN ${tag} ==="
  .venv-gpu-jax/bin/python src/run_long_integration_global.py \
    --mode-split --use-scan --dtype float32 \
    --kappa-gm 1000 --project-adv-vel --fct-adv \
    --seasonal-wind --wind-year 2023 --real-air-temp \
    --kappa-v 1e-6 --kappa-conv 0.01 \
    --lat-max 65 --ny 130 \
    --tag "${tag}" \
    --out-dir results/surface_heat_flux_65n \
    --log-dir logs/surface_heat_flux_65n \
    --days 365 --dt 3600 "$@"
}

run real_air_kv1e-6_kconv001_lat65_lambda20 --lambda-bulk 20
run real_air_kv1e-6_kconv001_lat65_lambda80 --lambda-bulk 80
