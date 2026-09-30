# Cross-process actual material restart replay

This is an engineering control for the already completed repaired 2-degree
float64, fixed-January-2023, no-ice trajectory. It does not introduce r-star
geometry or alter the production defaults. `protocol.json` freezes the gate:
the day-30 state, cumulative ledger and every thickness-history sample must
be byte-identical to the original continuous run, not merely close.

Use two separate CUDA Python processes, from the repository root:

```shell
python research/experiments/material_restart_replay/replay.py --start results/legacy_repair/m1_momentum_flux_C_20260929T152700Z/actual/strict_restart_target7d.npz --stop-day 15 --output results/replay_new/day7_to15
python research/experiments/material_restart_replay/replay.py --start results/replay_new/day7_to15/strict_restart.npz --stop-day 30 --output results/replay_new/day15_to30
```

Each output must be new. The runner checks the frozen actual source/input
hashes and reconstructs the strict contract from the original grid, parameters,
forcing manifest and protocol using the current factory. It requires the same
CUDA backend/device/JAX contract. It freezes its own inputs and sources,
stops at the first refused step without modifying the baseline, and archives
the attempted state and budget after asserting all-field rollback.

Missing local baseline/input data is an explicit refusal, not a request to
download replacements or reuse a different trajectory. This narrow real-data
replay cannot establish seasonal or century reliability, climate/forecast
skill, unsupported physics, or superiority to an industrial model.

After both processes terminate, inspect raw arrays independently of the solver
or restart loader:

```shell
python research/experiments/material_restart_replay/audit.py --replay-folder results/replay_new
```

The audit writes a new `validation_final.json` and refuses an existing output.
It verifies metadata and every raw array checksum, the clock/counter contract,
all history prefixes, reference state/ledger agreement, source/input identity,
the two-process checkpoint hash chain and final byte equality. Its corruption
controls are `tests/test_material_restart_raw_audit.py`.

The registered September-30 trial completed both numerical segments, but its
final cross-process byte gate **failed** at floating-point-tail scale. Source,
inputs, local strict round trips and actual source totals agree; five state
fields and parts of the cumulative ledger/history do not. The failed output
and independent audit remain archived. This runner has not established general
cross-process bit reproducibility, and its gate has not been relaxed.
