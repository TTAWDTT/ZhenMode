#!/usr/bin/env bash
# Stage-G ocean_solver side: monthly full-bulk surface forcing on the shared
# 0.5-degree grid. This is a pre-registered experiment and must not be treated
# as the baseline until the annual MOM6 gate is scored.
set -euo pipefail
cd "$(dirname "$0")/.."

DAYS="${DAYS:-30}"
SAVE_3D="${SAVE_3D:-false}"
SAVE_3D_TERMS="${SAVE_3D_TERMS:-false}"
INIT_FROM="${INIT_FROM:-}"
SMOOTH_PASSES="${SMOOTH_PASSES:-80}"
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
# Prefer the shared MOM6-derived 0.1-degree twin when it is available; this
# keeps Stage-G on the exact 67.9% ocean grid used by the external model.
STAGE_G_SHARED_TOPOG="${STAGE_G_SHARED_TOPOG:-/root/external_models/mom6_slice_050/p0_stage_f_dynamic_365d_v12/INPUT/topog_050.nc}"
SHARED_TOPOG_NPZ="data/stage_g/stage_g_shared_topog_010.nc.npz"
if [[ -z "${OCEAN_SOLVER_BATHYMETRY:-}" ]]; then
  if [[ -f "${STAGE_G_SHARED_TOPOG}" ]]; then
    "${PYTHON_BIN}" scripts/build_stage_g_shared_topog.py --topog "${STAGE_G_SHARED_TOPOG}" --out "${SHARED_TOPOG_NPZ}"
    export OCEAN_SOLVER_BATHYMETRY="${SHARED_TOPOG_NPZ%.npz}"
    SMOOTH_PASSES=0
  else
    export OCEAN_SOLVER_BATHYMETRY="/mnt/c/Users/zhen.luo/ocean_solver/data/ETOPO_2022_v1_r3600x1800_surface.nc"
  fi
fi
STAGE_G_WOA="${STAGE_G_WOA:-/root/external_models/mom6_slice_050/p0_stage_f_dynamic_365d_v12/INPUT/woa_ts_050.nc}"
STAGE_G_INIT="data/stage_g/stage_g_init_2023_050.npz"
if [[ -z "${INIT_FROM:-}" ]]; then
  if [[ -f "${STAGE_G_WOA}" ]]; then
    "${PYTHON_BIN}" scripts/build_stage_g_init_from_mom6_woa.py --woa "${STAGE_G_WOA}" --out "${STAGE_G_INIT}"
    INIT_FROM="${STAGE_G_INIT}"
  fi
fi
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
  --min-depth 500 --smooth-passes "${SMOOTH_PASSES}" \
  --nu-h 2e6 --dt 1800 \
  --days "${DAYS}" --snap-days "${SNAP_DAYS:-10}" \
  $( [[ -n "${INIT_FROM}" ]] && echo --init-from "${INIT_FROM}" ) \
  $( [[ "${SAVE_3D}" == "true" ]] && echo --save-3d ) \
  $( [[ "${SAVE_3D_TERMS}" == "true" ]] && echo --save-3d-terms ) \
  --tag "${TAG}" \
  --out-dir "${OUT_DIR}" \
  --log-dir "${LOG_DIR}"