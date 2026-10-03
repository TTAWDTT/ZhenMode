#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

LAMBDA="${1:-160}"
DAYS="${2:-30}"
TAU="${3:-30}"
CELLS="${4:-3}"
TAG="${5:-coastal_restore${TAU}_${DAYS}d}"
TAPER="${TAPER:-none}"

PYTHON_BIN="${PYTHON_BIN:-.venv-gpu-jax/bin/python}"
if [[ ! -x "${PYTHON_BIN}" && -x "/mnt/c/Users/zhen.luo/ocean_solver/.venv-gpu-jax/bin/python" ]]; then
  PYTHON_BIN="/mnt/c/Users/zhen.luo/ocean_solver/.venv-gpu-jax/bin/python"
fi
if [[ -d "/mnt/c/Users/zhen.luo/ocean_solver/data/woa" && -z "${OCEAN_SOLVER_WOA_DIR:-}" ]]; then
  export OCEAN_SOLVER_WOA_DIR="/mnt/c/Users/zhen.luo/ocean_solver/data/woa"
fi
export OCEAN_SOLVER_BATHYMETRY="${OCEAN_SOLVER_BATHYMETRY:-/mnt/c/Users/zhen.luo/Desktop/ETOPO_2022_v1_r3600x1800_surface.nc}"
OUT_DIR="${OUT_DIR:-results/coastal_restore_07}"
LOG_DIR="${LOG_DIR:-logs/coastal_restore_07}"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

"${PYTHON_BIN}" src/run_long_integration_global.py \
  --mode-split --use-scan --dtype float32 \
  --lat-max 65 --resolution 0.7 \
  --seasonal-wind --wind-year 2023 --real-air-temp \
  --lambda-bulk "${LAMBDA}" \
  --kappa-v 1e-6 --kappa-conv 0.01 --kappa-gm 0 \
  --localize-conv --fct-adv --project-adv-vel \
  --min-depth 500 --smooth-passes 80 \
  --coastal-restore-days "${TAU}" --coastal-restore-cells "${CELLS}" \
  --coastal-restore-taper "${TAPER}" \
  --days "${DAYS}" --dt 3600 \
  --tag "${TAG}" \
  --out-dir "${OUT_DIR}" \
  --log-dir "${LOG_DIR}"
