# Active external/internal comparison runs

Updated: 2026-09-27 13:25 local.

## Core direct comparison

| Run | Model | Status | Purpose |
|---|---|---|---|
| Stage-F exact dynamic bulk 30d | MOM6/ocean_solver | completed | monthly-air dynamic-bulk control |
| Stage-F exact dynamic bulk 365d v11 | MOM6 local | running | annual exact dynamic-bulk 3D comparison with 10-day temp+salt |
| ocean_solver Stage-F 365d no-ice | ocean_solver | completed | internal annual bulk control |
| ocean_solver Stage-F 365d 3D | ocean_solver | completed | annual 3D snapshot control |
| ocean_solver Stage-F 365d 3D latitude-depth/MLD | ocean_solver | completed | standardized upper-ocean error diagnostic |

## Recent annual gates

| Run | Status | Verdict |
|---|---|---|
| 365d stratification MLD, no ice | completed | not promoted |
| 365d northern-only stratification MLD, no ice | completed | not promoted |
| 365d northern-only stratification MLD, dynamic ice | completed | not promoted |
| 365d dynamic-ice/no-MLD ablation | completed | not promoted |
| 365d fixed 100m 40--60N MLD, no ice | completed | not promoted |
| 365d dynamic ice, no MLD, 3D | completed | not promoted; 3D/MLD gate fails |

## Completed 30d diagnostics

| Probe | Result |
|---|---|
| fine upper-z vertical grid | rejected; worse than default grid |
| low scalar mixing | rejected; essentially unchanged |
| stratification MLD, 40--60N | rejected on 30d regional gate |
| fixed 100m MLD, 40--60N | regional gain, but annual gate failed |
| dynamic ice + fixed 100m MLD, 40--60N | not annualized after annual fixed-depth failure |
| dynamic ice + stratification MLD | completed; 30d 3D diagnostic improves global RMSE, but annual candidate still under test |
| dynamic ice, 40--65N only | final-10d window: equivalent to full dynamic ice; no global/NA gain |



Note: v10 relaunched 2026-09-27 12:14 local with the exact v8 dynamic-bulk contract and daily temp+salt output. A 1-day salt smoke test passed, so this run enables the standardized MLD metric. It was alive at 12:48 local, but the WSL instance later became unavailable after the host D: volume filled. V11 was relaunched at 13:20 local with the exact v8 dynamic-bulk contract and 10-day temp+salt output; a 1-day smoke test validated the 10-day diag-table format. This reduces annual I/O enough to fit the available disk while still supporting the final-90d 3D and MLD gates.
