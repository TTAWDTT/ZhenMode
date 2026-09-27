#!/usr/bin/env bash
# Stage-G ocean_solver side: monthly full-bulk surface forcing on the shared
# 0.5-degree grid. This is a pre-registered experiment and must not be treated
# as the baseline until the annual MOM6 gate is scored.
set -euo pipefail
cd "$(dirname "$0")/.."

DAYS="${DAYS:-30}"
SAVE_3D="${SAVE_3D:-false}"
SAVE_3D_TERMS="${SAVE_3D_TERMS:-false}"
TAG="${1:-industrial_comparison_050_stage_g_${DAYS}d}"
OUT_DIR="${OUT_DIR:-results/industrial_comparison_045}"
LOG_DIR="${LOG_DIR:-logs/industrial_comparison_045}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/c/Users/zhen.luo/ocean_solver/.venv-gpu-jax/bin/python}"
if [[ ! -x "${PYTHON_BIN}" && -x "/mnt/c/Users/zhen.luo/ocean_solver/.venv-gpu-jax/bin/python" ]]; then
  PYTHON_BIN="/mnt/c/Users/zhen.luo/ocean_solver/.venv-gpu-jax/bin/python"
fi
if [[ -d "/mnt/c/Users/zhen.luo/ocean_solver/data/woa" && -z "${OCEAN_SOLVER_WOA_DIR:-}" ]]; then
  export OCEAN_SOLVER_WOA_DIR="/mnt/c/Users/zhen.luo/ocean_solver/data/woa"
fi
export OCEAN_SOLVER_BATHYMETRY="${OCEAN_SOLVER_BATHYMETRY:-/mnt/c/Users/zhen.luo/Desktop/ETOPO_2022_v1_r3600x1800_surface.nc}"
STAGE_G_FORCING="${STAGE_G_FORCING:-data/stage_g/stage_g_forcing_2023_050.npz}"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

"${PYTHON_BIN}" src/run_long_integration_global.py \
  --mode-split --use-scan --dtype float32 \
  --lat-max 65 --resolution 0.5 --resolution-remap area \
  --seasonal-wind --wind-year 2023 \
  --full-bulk \
  --full-bulk-forcing "${STAGE_G_FORCING}" \
  --no-meridional-heat-flux \
  --no-bulk-flux \
  --kappa-v 1e-6 --kappa-conv 0.01 \
  --kappa-gm 0 \
  --localize-conv --fct-adv --project-adv-vel \
  --min-depth 500 --smooth-passes 80 \
  --nu-h 2e6 --dt 1800 \
  --days "${DAYS}" --snap-days "${SNAP_DAYS:-10}" \
  $( [[ "${SAVE_3D}" == "true" ]] && echo --save-3d ) \
  $( [[ "${SAVE_3D_TERMS}" == "true" ]] && echo --save-3d-terms ) \
  --tag "${TAG}" \
  --out-dir "${OUT_DIR}" \
  --log-dir "${LOG_DIR}"