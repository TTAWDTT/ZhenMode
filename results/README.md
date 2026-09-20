# `results/` — run outputs and figure provenance

This directory is **gitignored** (`results/*`, with this file re-included).
It is where the driver writes its outputs:

```
global_<tag>.npz       2D monitor tables + final snapshot (run_long_integration_global)
global_<tag>_3d/       streamed 3D T/U/V snapshots, one .npy per save
<tag>_analysis.npz     derived series (see the _*.py helpers)
```

A handful of files under `results/` are **deliberately tracked** because they
are the evidence behind figures cited in the docs. Everything else here is
scratch and can be deleted. The run logs live in `logs/`, not here.

## Tracked data files

| File | Consumed by |
| --- | --- |
| `sverdrup_global_g365d_013.npz` | `campaign_compare/make_fig10_sverdrup.py` |
| `campaign_compare/phys_check_g365d_013.npz` | `campaign_compare/make_fig12_physical_exam.py` |
| `campaign_compare/moc2_g365d_013.npz` | `campaign_compare/make_fig12_physical_exam.py` |
| `campaign_compare/tenyr_g3650d_014.json`, `..._016.json` | 10-year AMOC/GMOC/SSS series; kept as evidence (no script currently reads them) |

The `.png` files under `campaign_compare/` are the published figures;
`campaign_compare/FIGCAPTIONS_zh.md` captions them.

## Scripts

None of these are part of the model or of the test suite. They are
**archival provenance**: they record how each figure / number was produced.
Most need run outputs that are *not* in the repository (they live on the GPU
node or were pulled to the author's machine), so they are not reproducible
from a fresh clone.

**Cluster-side** — hardcode `/data/tmp/ocean` and are meant to be run on the
GPU node, not here:

- `_curves_from_drift.py` — rebuild per-tag analysis series from drift tables
- `_sverdrup_remote.py` — depth-integrated Sverdrup transport
- `_spinup_probe_analysis.py` — deep-T / AMOC verdict for a spin-up probe

**Local** — run from the repository root on the author's machine:

- `_smoke_v3.py` — smoke harness for the analysis pipeline
- `_moc_ms.py` — AMOC / deep-T analysis for a 10-year mode-split run
- `_probe_sss_restore.py` — SSS restoring probe
- `acceptance_g365d_012/make_acceptance_figs.py`
- `campaign_compare/make_*.py` — the campaign figures. `make_campaign_figs.py`
  also reads `C:/Users/zhen.luo/.research` (machine-specific console logs).
