#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/zhen.luo/.codex/worktrees/3f4d/ocean_solver

PYTHON_BIN="/mnt/c/Users/zhen.luo/ocean_solver/.venv-gpu-jax/bin/python"
export OCEAN_SOLVER_BATHYMETRY="/mnt/c/Users/zhen.luo/Desktop/ETOPO_2022_v1_r3600x1800_surface.nc"
export OCEAN_SOLVER_WOA_DIR="/mnt/c/Users/zhen.luo/ocean_solver/data/woa"
mkdir -p results/gm0_attribution_07 logs/gm0_attribution_07

run() {
  local tag="$1"; local kappa="$2"
  echo "=== RUN ${tag} ==="
  "${PYTHON_BIN}" src/run_long_integration_global.py \
    --mode-split --use-scan --dtype float32 \
    --lat-max 65 --resolution 0.7 \
    --seasonal-wind --wind-year 2023 --real-air-temp \
    --lambda-bulk 80 --kappa-v 1e-6 --kappa-conv 0.01 \
    --kappa-gm "${kappa}" \
    --localize-conv --fct-adv --project-adv-vel \
    --days 365 --dt 3600 --snap-days 30 \
    --save-3d --save-3d-terms \
    --tag "${tag}" \
    --out-dir results/gm0_attribution_07 \
    --log-dir logs/gm0_attribution_07
}

run gm0_3dterms 0
run gm500_3dterms 500
