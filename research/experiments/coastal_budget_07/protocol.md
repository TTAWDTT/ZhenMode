# Coastal Surface Heat Budget Protocol

Status: complete
Date: 2026-09-23
Question: Which surface heat-budget term does the diagnostic coastal SST
constraint replace?

Runs:
1. 30d baseline with the all-around diagnostic candidate physics.
2. 30d identical run plus hard coastal SST restore, `tau=0.5d`, cells<=7.
Both runs store 3D tracer terms and 3D snapshots. The budget averages day 20
and 30 snapshots and uses cell-area weights.

Metric: area-weighted top-layer tendency in K/day for advection, horizontal
diffusion, vertical diffusion, convection, bulk flux, and coastal SST restore.

Decision rule: identify whether the restore benefit is consistent with missing
local diffusion, missing bulk exchange, or a missing boundary-value/lateral
transport closure.
