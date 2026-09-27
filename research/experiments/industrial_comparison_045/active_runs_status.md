# Active external/internal comparison runs

Updated: 2026-09-27 12:10 local.

## Core direct comparison

| Run | Model | Status | Purpose |
|---|---|---|---|
| Stage-F exact dynamic bulk 30d | MOM6/ocean_solver | completed | monthly-air dynamic-bulk control |
| Stage-F exact dynamic bulk 365d v9 | MOM6 local | running | fresh annual exact dynamic-bulk 3D comparison after v8 was lost; daily temp only |
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



Note: v9 relaunched 2026-09-27 12:02 local with the exact v8 dynamic-bulk contract. The first attempt to reduce diag output to 10-day cadence was rejected by the installed MOM6 diag-table parser, so v9 uses the original daily temp-only contract. Salt MLD is therefore unavailable from this run. It is held by a persistent WSL session and reached 4.75 simulated days in 5.9 wall minutes at 12:10 local.

