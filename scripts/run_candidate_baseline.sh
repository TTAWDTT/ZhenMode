#!/usr/bin/env bash
# Locked diagnostic baseline: 65N, 0.7 degree, annual real air, lambda80,
# reduced vertical mixing, localized convection, GM0, FCT/TVD transport.
set -euo pipefail
cd "$(dirname "$0")/.."

export OCEAN_SOLVER_BATHYMETRY="${OCEAN_SOLVER_BATHYMETRY:-/mnt/c/Users/zhen.luo/Desktop/ETOPO_2022_v1_r3600x1800_surface.nc}"
OUT_DIR="${OUT_DIR:-results/candidate_65n_07_gm0}"
LOG_DIR="${LOG_DIR:-logs/candidate_65n_07_gm0}"
TAG="${1:-candidate_65n_07_gm0}"
DAYS="${DAYS:-365}"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

.venv-gpu-jax/bin/python src/run_long_integration_global.py \
  --mode-split --use-scan --dtype float32 \
  --lat-max 65 --resolution 0.7 \
  --seasonal-wind --wind-year 2023 --real-air-temp \
  --lambda-bulk 80 \
  --kappa-v 1e-6 --kappa-conv 0.01 \
  --kappa-gm 0 \
  --localize-conv --fct-adv --project-adv-vel \
  --days "${DAYS}" --dt 3600 \
  --tag "${TAG}" \
  --out-dir "${OUT_DIR}" \
  --log-dir "${LOG_DIR}"
