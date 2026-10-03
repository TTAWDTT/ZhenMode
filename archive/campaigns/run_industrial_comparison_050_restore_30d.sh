#!/usr/bin/env bash
# Stage R ocean_solver side: same dynamic core as the MOM6 wind-only slice,
# but with prescribed WOA surface temperature and salinity restoring.
set -euo pipefail
cd "$(dirname "$0")/.."

DAYS="${DAYS:-30}"
SST_RESTORE_DAYS="${SST_RESTORE_DAYS:-30}"
SSS_RESTORE_DAYS="${SSS_RESTORE_DAYS:-30}"
TAG="${1:-industrial_comparison_050_restore_${DAYS}d}"
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
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

"${PYTHON_BIN}" src/run_long_integration_global.py \
  --mode-split --use-scan --dtype float32 \
  --lat-max 65 --resolution 0.5 --resolution-remap area \
  --seasonal-wind --wind-year 2023 \
  --no-bulk-flux --no-meridional-heat-flux \
  --lambda-bulk 80 \
  --global-sst-restore-days "${SST_RESTORE_DAYS}" \
  --sss-restore-days "${SSS_RESTORE_DAYS}" \
  --kappa-v 1e-6 --kappa-conv 0.01 \
  --kappa-gm 0 \
  --localize-conv --fct-adv --project-adv-vel \
  --min-depth 500 --smooth-passes 80 \
  --nu-h 2e6 --dt 1800 \
  --days "${DAYS}" --snap-days "${SNAP_DAYS:-10}" \
  --tag "${TAG}" \
  --out-dir "${OUT_DIR}" \
  --log-dir "${LOG_DIR}"
