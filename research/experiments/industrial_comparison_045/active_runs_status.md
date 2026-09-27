# Active external/internal comparison runs

Updated: 2026-09-27 13:48 local.

## Core direct comparison

| Run | Model | Status | Purpose |
|---|---|---|---|
| Stage-F exact dynamic bulk 30d | MOM6/ocean_solver | completed | monthly-air dynamic-bulk control |
| Stage-F exact dynamic bulk 365d v11 | MOM6 local | running | annual exact dynamic-bulk 3D comparison with 10-day temp+salt |
| ocean_solver Stage-F 365d no-ice | ocean_solver | completed | internal annual bulk control |
| ocean_solver Stage-F 365d 3D | ocean_solver | completed | annual 3D snapshot control |
| ocean_solver Stage-F 365d 3D latitude-depth/MLD | ocean_solver | completed | standardized upper-ocean error diagnostic |
| ocean_solver Stage-F band ice + 100m MLD 30d | ocean_solver | completed | passed all pre-registered 30d gates |
| ocean_solver Stage-F band ice + 100m MLD 365d v1 | ocean_solver local | running | annual final-90d 3D/MLD gate |

## Recent annual gates

| Run | Status | Verdict |
|---|---|---|
| 365d stratification MLD, no ice | completed | not promoted |
| 365d northern-only stratification MLD, no ice | completed | not promoted |
| 365d northern-only stratification MLD, dynamic ice | completed | not promoted |
| 365d dynamic-ice/no-MLD ablation | completed | not promoted |
| 365d fixed 100m 40--60N MLD, no ice | completed | not promoted |
| 365d dynamic ice, no MLD, 3D | completed | not promoted; 3D/MLD gate fails |
| 365d dynamic ice 40--65N + fixed 100m MLD 40--60N | running | 30d gate passed; annual gate pending |

## Completed 30d diagnostics

| Probe | Result |
|---|---|
| fine upper-z vertical grid | rejected; worse than default grid |
| low scalar mixing | rejected; essentially unchanged |
| stratification MLD, 40--60N | rejected on 30d regional gate |
| fixed 100m MLD, 40--60N | regional gain, but annual gate failed |
| dynamic ice + fixed 100m MLD, 40--60N | passed the corrected final-10d 30d gate; annualized for a check |
| dynamic ice + stratification MLD | rejected on corrected final-10d window |
| dynamic ice, 40--65N only | final-10d window: equivalent to full dynamic ice; no global/NA gain |

## Notes

- MOM6 v11 uses the exact dynamic-bulk contract and 10-day temperature/salinity
  output to support the final-90d 3D and MLD gates while limiting I/O.
- The new ocean_solver annual check uses the same forcing contract, 30-day
  3D snapshots, and final-90d scoring window.
- Host C: and D: remain nearly full; obsolete WSL-side diagnostic snapshots were
  removed after their benchmark JSONs were retained.
