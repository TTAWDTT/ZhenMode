# Active external/internal comparison runs

Updated: 2026-09-27 01:41 local.

All long runs below were relaunched after a WSL restart at 01:40 local.

| Run | Model | Status | Purpose |
|---|---|---|---|
| wind 30d | MOM6/ocean_solver | completed | first dynamic-control comparison |
| restore 30d | MOM6/ocean_solver | completed | prescribed-restore comparison |
| restore 365d | MOM6 local | running | first annual Stage-R comparison |
| Stage-F exact dynamic bulk 30d | MOM6/ocean_solver | completed | exact monthly-air bulk control |
| Stage-F exact dynamic bulk 365d | MOM6 local | running | annual exact dynamic-bulk comparison |
| ocean_solver Stage-F 365d no-ice | ocean_solver | completed | internal annual bulk control |
| ocean_solver Stage-F 365d MLD20 | ocean_solver | completed | stratification sensitivity; not promoted |
| Stage-I annual dynamic ice | ocean_solver | completed | minimal ice/MLD closed loop |
| ocean_solver Stage-F 365d 3D | ocean_solver | running | annual 3D snapshot counterpart for MOM6 bulk comparison |
