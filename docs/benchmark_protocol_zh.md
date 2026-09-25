# Standardized Benchmark Protocol

日期：2026-09-26  
状态：draft  
目的：把当前内部 A/B 实验放进可复现、可外推、可对接工业级模式讨论的坐标系。

## 目标

1. 固定数据、指标和运行方式，避免“每换一个 idea 就换一套 score”。
2. 把模型能力放在与 MOM6 / NEMO / FESOM2 / ICON-Ocean 等成熟模式可对话的语境里。
3. 为后续 sea-ice / mixed-layer / boundary closure 的改进提供可判断的验收线。

## 1. Baseline

Current production-like candidate:

- `candidate_65n_045_icefloor`
- 0.45°, 65N, 365d
- `dt=1800s`, `nu_h=2e6`, `min_depth=500`, `smooth_passes=80`
- lambda_bulk=80
- `kappa_v=1e-6`
- localized convection
- FCT/TVD transport
- annual NCEP R1 2m air forcing
- seasonal NCEP wind
- ice-air floor proxy at `-1.8 C`

Command:

```bash
bash scripts/run_candidate_baseline_045_icefloor.sh
```

Reference result:

```text
results/ice_proxy_045/global_res045_icefloor_365d_repeat.npz
```

Scored metrics:

```text
research/experiments/ice_proxy_045/benchmark_365d_repeat.json
```

## 2. Required run setup

For any run entered into the benchmark table:

1. 0.45° horizontal resolution.
2. 65N domain.
3. 365d integration.
4. seasonal NCEP wind.
5. annual real 2m air forcing.
6. lambda_bulk=80.
7. kappa_v=1e-6.
8. localized convective adjustment.
9. FCT/TVD transport.
10. projected stage-2 advective velocity.
11. ice-air floor proxy at -1.8 C.
12. `PASS` verdict.
13. Reproducible repeat or restart.

This is deliberately strict: a run that only improves one number after a new
physical tuning knob is not accepted as a candidate.

## 3. Primary metrics

Report all of the following, not just the best-looking metric:

| Metric | Definition |
|---|---|
| `global A2 RMSE` | 2-degree-smoothed SST RMSE vs WOA |
| `NA 40..60N A2 RMSE` | regional 2-degree-smoothed SST RMSE |
| `near-wall 55..60N bias` | raw coastal bias |
| `raw global bias` | area-unweighted but ocean-only mean error |
| `ice extent` | area of cells below freezing point |
| `heat drift` | relative heat-content drift |
| `salt drift` | relative salt-content drift |
| `verdict` | solver stability watchdog result |
| `wall time` | integration cost |

For MLD / mixed-layer experiments, additionally report:

- mean mixed-layer depth
- median mixed-layer depth
- 90th-percentile MLD
- MLD bias/rmse vs the same definition applied to WOA
- fraction of columns with MLD shallower than 20 m

The current preferred MLD definition is density-threshold MLD using the
solver's linear EOS, with a `0.03 kg/m3` density delta from a 10 m reference
depth.

## 4. Sea-ice / mixed-layer minimal closure

The minimum closure should be a real coupled thermodynamic closure, not a
pure post-hoc mask.  The first implementation should support:

1. **Mixed-layer heat capacity**
   - surface heat flux spread over a diagnosed or prescribed mixed-layer depth;
   - not just the 5 m top grid cell;
2. **Thermodynamic sea-ice proxy**
   - freezing-point temperature;
   - ice-covered surface flux damping;
   - brine-rejection salt flux;
   - ice thickness or effective ice state;
   - melt/growth feedback.

   The current solver coupling has two opt-in switches:

   ```text
   --mixed-layer-depth 50.0
   --ice-salt-flux 1e-7 --ice-freeze-temp -1.8
   ```

   `--mixed-layer-depth` spreads the surface heat flux over a well-mixed
   slab.  `--ice-salt-flux` adds brine rejection only where the live SST is at
   or below the freezing point.  Both are off by default.
3. **Diagnostics**
   - mixed-layer depth
   - sea-ice extent
   - ocean-to-ice heat flux
   - salt flux from freezing/melting
   - surface heat budget residual

The first target is a "minimum closed loop":

> atmospheric forcing -> mixed-layer heat capacity -> sea-ice formation/melt -> ocean heat/salt flux -> SST/MLD response.

It is acceptable that this is simpler than CICE, SI3, or Icepack.  It is not
acceptable to call it a full sea-ice model until thickness and brine rejection
are actually represented.

## 5. Reproducibility

Every benchmark run must record:

- exact command line
- git commit
- random/forcing year
- resolution
- grid remap
- dt
- physics flags
- closure parameters
- initial/reference fields
- hardware if performance is reported

The minimum acceptance test is:

```bash
python -m pytest tests -q
python -m ruff check src scripts
python src/benchmark_metrics.py --npz <run.npz> --out <run>_benchmark.json
```

A candidate is accepted only if:

1. stability verdict is `PASS`;
2. heat/salt drift is within the pre-registered tolerance;
3. global A2 is not worse than the locked baseline;
4. NA 40..60N A2 is not worse;
5. near-wall raw bias is not worse;
6. the result is reproducible within numerical precision.

### Standardized manifest and comparison table

To keep a run reproducible and portable, write a manifest after scoring:

```bash
python src/benchmark_manifest.py \
  --npz <run.npz> \
  --metrics <run>_benchmark.json \
  --commit <git-commit> \
  --model <model-name> \
  --config-json <config.json> \
  --out <run>_manifest.json
```

To compare several benchmark JSON files without editing reports by hand:

```bash
python src/benchmark_table.py \
  <benchmark-a.json> <benchmark-b.json> \
  --label "A=<benchmark-a.json>" \
  --label "B=<benchmark-b.json>"
```

The table currently reports verdict, duration, global A2, North Atlantic RMSE,
near-wall RMSE, global bias, and heat/salt drift.
## 6. Industrial-mode comparison stance

We should not claim that `ocean_solver` surpasses industrial-grade models as
full systems.  The realistic target is to become competitive on specific,
well-defined slices:

| Slice | Meaning |
|---|---|
| SST climate slice | same forcing, same grid, same reference field |
| coastal bias slice | near-wall/regional RMSE and bias |
| mixed-layer slice | MLD and sea-ice extent diagnostics |
| GPU throughput | wall time per simulation year |
| reproducibility | deterministic repeat and CI-clean code |

The first realistic target is not to beat MOM6/NEMO globally, but to produce a
benchmarked result that is internally consistent, externally interpretable, and
clearly competitive on the selected slice.

---

Reference implementation:

```bash
python src/benchmark_metrics.py \
  --npz results/ice_proxy_045/global_res045_icefloor_365d_repeat.npz \
  --out research/experiments/ice_proxy_045/benchmark_365d_repeat.json
```

Current baseline benchmark is stored as
`research/experiments/ice_proxy_045/benchmark_365d_repeat.json`.

The reusable scorer emits A1/A2, regional, ice, heat/salt drift and
stability fields.  The current repeat reports:

- verdict `PASS`
- global A2 RMSE `1.113 C`
- A1 corr `0.9991`
- A2 corr `0.9955`
- raw global bias `-0.538 C`
- raw global RMSE `0.914 C`
- zero below-freezing surface cells
- heat drift `-0.283%`
- salt drift `-0.00036%`
- mean MLD `31.5 m` (density-threshold definition)



