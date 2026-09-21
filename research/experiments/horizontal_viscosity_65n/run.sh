#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/zhen.luo/ocean_solver
export OCEAN_SOLVER_BATHYMETRY=/mnt/c/Users/zhen.luo/Desktop/ETOPO_2022_v1_r3600x1800_surface.nc
mkdir -p results/horizontal_viscosity_65n logs/horizontal_viscosity_65n

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
    --out-dir results/horizontal_viscosity_65n \
    --log-dir logs/horizontal_viscosity_65n \
    --days 365 --dt 3600 "$@"
}

run real_air_lambda80_gm500_localconv_nuh2p5e6 --nu-h 2.5e6
run real_air_lambda80_gm500_localconv_nuh1e6 --nu-h 1e6
