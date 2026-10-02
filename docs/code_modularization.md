# Production modularization

## 2026-10-02 — first extraction

The immutable comparison source is main `4bcb3e4002a4b276ac740d5768eb26d965bb77e1`. Historical manifests and research reports remain at their original source identity.

The first extraction moves FD state/parameter types, geometry construction, horizontal and vertical operators, EOS, pressure, transport, projection, and CFL calculation into `ocean_solver.fd`. Pure grid types and nodal thickness live in `ocean_solver.geometry`; the budget schema is independent of solver assembly in `ocean_solver.audit.schema`. Legacy flat modules directly re-export the same objects. Namedtuple field order, defaults, pickle module names, and JAX pytree behavior are preserved.

A fresh source registry includes every new package module. Current production and material restart contracts hash actual relocated execution files; a modified or missing new leaf is rejected. Earlier strict-source restarts require their historical commit. No hash is substituted to make old contracts appear compatible.

Validation uses the project-local inherited Python 3.12.14 environment, JAX/JAXlib 0.11.2, NumPy 2.5.3, CPU backend. The bounded runner enforces one CPU, 180 seconds, and 4 GiB per invocation. `scripts/capture_modularization_state.py` records a synthetic nonzero 8×4×6 full-state run, independently audited shadow states/ledgers, rejected-state rollback, restart round-trip, factory arities, and operator outputs. This is a refactoring equivalence witness, not a physical qualification or accuracy claim.

| Check | Immutable reference | First extraction |
| --- | --- | --- |
| Full numerical/interface payload | 275 records | All 275 have identical dtype, shape, and bytes |
| Bounded capture wall time | 30.438 s | 33.343 s |
| Peak interpreter RSS | 461,897,728 bytes | 461,189,120 bytes |
| Peak job private memory | 441,151,488 bytes | 440,557,568 bytes |

Representative real JAX heat ledger, genuine internal diffusion fault, and invalid-timestep refusal checks passed before and after (6 baseline tests; 9 after including type/API and packaging checks). Three dedicated modularization contracts passed, including changed-source restart refusal and missing-source refusal. The existing environment emitted a NumPy binary-size warning on the restart fixture; numerical captures and checks completed successfully.

Reproduction from the repository root (local environment with the above dependencies):

```powershell
python scripts/run_bounded_research_tests.py --module scripts.capture_modularization_state --source-root src --output logs/code_cleanup/current_legacy.npz
python scripts/run_bounded_research_tests.py tests/test_modularization_contracts.py -q
```

Reference arrays and source snapshots are ignored local validation artifacts. No private model input, archive array, filesystem path, or secret is committed. The default production method and CLI remain unchanged. Driver orchestration and the remaining solver process/step/factory decomposition are subsequent units; this entry does not claim that larger refactor is finished.
