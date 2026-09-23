#!/usr/bin/env bash
# Candidate 0.45-degree ice-floor baseline: 65N, lambda80, GM0,
# localized convection, FCT/TVD transport, annual real air forcing.
set -euo pipefail
cd "$(dirname "$0")/.."

LAMBDA="${LAMBDA:-80}"
DAYS="${DAYS:-365}"
TAG="${1:-candidate_65n_045_icefloor_${DAYS}d}"

PYTHON_BIN="${PYTHON_BIN:-.venv-gpu-jax/bin/python}"
if [[ ! -x "${PYTHON_BIN}" && -x "/mnt/c/Users/zhen.luo/ocean_solver/.venv-gpu-jax/bin/python" ]]; then
  PYTHON_BIN="/mnt/c/Users/zhen.luo/ocean_solver/.venv-gpu-jax/bin/python"
fi
if [[ -d "/mnt/c/Users/zhen.luo/ocean_solver/data/woa" && -z "${OCEAN_SOLVER_WOA_DIR:-}" ]]; then
  export OCEAN_SOLVER_WOA_DIR="/mnt/c/Users/zhen.luo/ocean_solver/data/woa"
fi
export OCEAN_SOLVER_BATHYMETRY="${OCEAN_SOLVER_BATHYMETRY:-/mnt/c/Users/zhen.luo/Desktop/ETOPO_2022_v1_r3600x1800_surface.nc}"
OUT_DIR="${OUT_DIR:-results/candidate_65n_045_icefloor}"
LOG_DIR="${LOG_DIR:-logs/candidate_65n_045_icefloor}"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

"${PYTHON_BIN}" src/run_long_integration_global.py \
  --mode-split --use-scan --dtype float32 \
  --lat-max 65 --resolution 0.45 --resolution-remap area \
  --seasonal-wind --wind-year 2023 --real-air-temp --ice-air-floor \
  --lambda-bulk "${LAMBDA}" \
  --kappa-v 1e-6 --kappa-conv 0.01 \
  --kappa-gm 0 \
  --localize-conv --fct-adv --project-adv-vel \
  --min-depth 500 --smooth-passes 80 \
  --nu-h 2e6 --dt 1800 \
  --days "${DAYS}" \
  --tag "${TAG}" \
  --out-dir "${OUT_DIR}" \
  --log-dir "${LOG_DIR}"



