# Sub-Freezing Air-Forcing Proxy at 0.45 Degree

Status: active
Date: 2026-09-24

## Physical issue

The 0.45-degree 365d GM0 candidate reaches a final SST minimum of about
`-3.03 C`, with `3030` ocean cells below the nominal freezing point. The WOA
reference never goes below about `-1.46 C`. The current bulk target therefore
forces open water below freezing.

## Proxy

Add an opt-in `--ice-air-floor`. It floors the bulk target at
`-1.8 C` where the observed 2m air is colder. This is a simple sea-ice/freezing
surface proxy, not a full ice model: it prevents sub-freezing forcing but does
not represent ice transport, thickness, brine rejection, or albedo.

## First 30d A/B

| run | global A2 | NA 40--60N A2 | near-wall raw bias | min SST |
|---|---:|---:|---:|---:|
| GM0 control | 0.9190 C | 0.8494 C | -0.7425 C | below freezing |
| ice-air floor | 0.9041 C | 0.8494 C | -0.7426 C | `-1.22 C` |

The ice proxy improves global A2 by `1.63%`, leaves North Atlantic metrics
essentially unchanged, and removes all below-freezing cells. Promote it to a
365d stability/climate check.

## Decision rules

- Reject if the annual run fails or loses the global improvement.
- Do not couple to another closure in this first test.
- If it improves, reproduce before promotion; if it is neutral at 365d, keep it
  as a physical plausibility fix but not a performance candidate.
