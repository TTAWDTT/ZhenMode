# Coastal T Restoring Taper Protocol

Status: complete
Date: 2026-09-23
Question: Can a tapered coastal SST restoring profile improve on the hard
0..7-cell diagnostic band?

Primary control: hard-band restore with `tau=0.5d`, cells<=7.
Primary metrics: 30d global A2, North Atlantic 40..60N A2, and near-wall
55..60N A2 RMSE.

Runs:
1. cosine taper, `tau=0.5d`, cells<=7.
2. cosine taper, `tau=0.25d`, cells<=7, to compensate for the reduced
   integrated restoring weight.

Decision rule: run 365d only if the 30d probe improves all three metrics over
the hard-band control.

Result: both taper variants fail. The natural cosine taper is worse on every
metric. The equal-strength cosine improves global A2 slightly over the natural
taper but is still worse than the hard band and has a larger near-wall RMSE.
