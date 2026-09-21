#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/zhen.luo/ocean_solver
export OCEAN_SOLVER_BATHYMETRY=/mnt/c/Users/zhen.luo/Desktop/ETOPO_2022_v1_r3600x1800_surface.nc
mkdir -p results/heat_tendency_decomposition logs/heat_tendency_decomposition

.venv-gpu-jax/bin/python src/run_long_integration_global.py \
  --mode-split --use-scan --dtype float32 \
  --project-adv-vel --fct-adv \
  --seasonal-wind --wind-year 2023 --real-air-temp \
  --lambda-bulk 80 --kappa-gm 500 --kappa-v 1e-6 --kappa-conv 0.01 \
  --localize-conv \
  --lat-max 65 --ny 130 \
  --days 365 --dt 3600 --snap-days 30 \
  --save-3d --save-3d-terms \
  --tag real_air_lambda80_gm500_localconv_3dterms \
  --out-dir results/heat_tendency_decomposition \
  --log-dir logs/heat_tendency_decomposition
