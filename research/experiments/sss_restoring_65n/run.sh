#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/zhen.luo/ocean_solver
export OCEAN_SOLVER_BATHYMETRY=/mnt/c/Users/zhen.luo/Desktop/ETOPO_2022_v1_r3600x1800_surface.nc
mkdir -p results/sss_restoring_65n logs/sss_restoring_65n

run() {
  local tag="$1"; shift
  echo "=== RUN ${tag} ==="
  .venv-gpu-jax/bin/python src/run_long_integration_global.py \
    --mode-split --use-scan --dtype float32 \
    --project-adv-vel --fct-adv \
    --seasonal-wind --wind-year 2023 --real-air-temp \
    --lambda-bulk 80 --kappa-gm 500 --kappa-v 1e-6 --kappa-conv 0.01 \
    --localize-conv \
    --lat-max 65 --ny 130 \
    --tag "${tag}" \
    --out-dir results/sss_restoring_65n \
    --log-dir logs/sss_restoring_65n \
    --days 365 --dt 3600 "$@"
}

run real_air_lambda80_gm500_localconv_sss30 --sss-restore-days 30
run real_air_lambda80_gm500_localconv_sss90 --sss-restore-days 90
