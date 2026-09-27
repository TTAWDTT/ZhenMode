# Active external/internal comparison runs

Updated: 2026-09-27 18:20 local.

## Core direct comparison

| Run | Model | Status | Purpose |
|---|---|---|---|
| Stage-F exact dynamic bulk 30d | MOM6/ocean_solver | completed | monthly-air dynamic-bulk control |
| Stage-F exact dynamic bulk 365d v11 | MOM6 local | killed before completion | earlier annual direct-comparison attempt |
| Stage-F exact dynamic bulk 365d v12 | MOM6 local | running | annual exact dynamic-bulk 3D comparison |
| ocean_solver Stage-F 365d no-ice | ocean_solver | completed | internal annual bulk control |
| ocean_solver Stage-F 365d 3D | ocean_solver | completed | annual 3D snapshot control |
| ocean_solver Stage-F 365d 3D latitude-depth/MLD | ocean_solver | completed | standardized upper-ocean error diagnostic |
| ocean_solver Stage-F band ice + 100m MLD 30d | ocean_solver | completed | passed the valid 30d gate |
| ocean_solver Stage-F band ice + 100m MLD 365d v2 | ocean_solver local | completed | annual climate gate failed |
| ocean_solver Stage-F band ice + 100m MLD + cooling gate 365d v2 | ocean_solver local | completed | annual climate gate rejected |

## Recent annual gates

| Run | Status | Verdict |
|---|---|---|
| 365d stratification MLD, no ice | completed | not promoted |
| 365d northern-only stratification MLD, no ice | completed | not promoted |
| 365d northern-only stratification MLD, dynamic ice | completed | not promoted |
| 365d dynamic-ice/no-MLD ablation | completed | not promoted |
| 365d fixed 100m 40--60N MLD, no ice | completed | not promoted |
| 365d dynamic ice, no MLD, 3D | completed | not promoted; 3D/MLD gate fails |
| 365d dynamic ice 40--65N + fixed 100m MLD 40--60N v1 | completed | invalid: bathymetry mismatch |
| 365d dynamic ice 40--65N + fixed 100m MLD 40--60N v2 | completed | rejected on annual climate/3D gate |

## Completed 30d diagnostics

| Probe | Result |
|---|---|
| fine upper-z vertical grid | rejected; worse than default grid |
| low scalar mixing | rejected; essentially unchanged |
| stratification MLD, 40--60N | rejected on 30d regional gate |
| fixed 100m MLD, 40--60N | regional gain, but annual gate failed |
| dynamic ice + fixed 100m MLD, 40--60N | passed the corrected final-10d 30d gate; rejected on valid annual rerun |
| dynamic ice + stratification MLD | rejected on corrected final-10d window |
| dynamic ice, 40--65N only | final-10d window: equivalent to full dynamic ice; no global/NA gain |

## Notes

- The 30d probe and annual no-ice control both use 67.9% ocean; the invalid v1
  used 90.8% ocean from a different bathymetry file.
- MOM6 v12 uses the same 365d Stage-F exact dynamic-bulk setup as v11 and saves
  10-day temperature/salinity output for the final-90d 3D/MLD gate.
- The active ocean_solver annual candidate is the cooling-gate variant of the
  band-ice/fixed-MLD run; it is not promoted until the annual gate passes.
- The paired annual manifests now pass the standardized contract validator, with
  provenance explicitly marked not_comparable because the exact source commits
  were not recorded with the saved outputs.

