# Sub-Freezing Air-Forcing Proxy at 0.45 Degree

Status: reproducibility
Date: 2026-09-24

## Physical issue

The 0.45-degree 365d GM0 candidate reaches a final SST minimum of about
`-3.03 C`, with `3030` ocean cells below the nominal freezing point. The WOA
reference never goes below about `-1.46 C`. The current bulk target therefore
forces open water below freezing.

## Proxy

Add an opt-in `--ice-air-floor`. It floors the bulk target at `-1.8 C` where
observed 2m air is colder. This is a simple sea-ice/freezing surface proxy, not
a full ice model: it prevents sub-freezing forcing but does not represent ice
transport, thickness, brine rejection, or albedo.

## 30d A/B

| run | global A2 | NA 40--60N A2 | near-wall raw bias | min SST |
|---|---:|---:|---:|---:|
| GM0 control | 0.9190 C | 0.8494 C | -0.7425 C | below freezing |
| ice-air floor | 0.9041 C | 0.8494 C | -0.7426 C | `-1.22 C` |

The ice proxy improves global A2 by `1.63%`, leaves North Atlantic metrics
essentially unchanged, and removes all below-freezing cells.

## 365d validation

| run | global A2 | NA 40--60N A2 | near-wall raw bias | below-freezing cells |
|---|---:|---:|---:|---:|
| GM0 control | 1.1488 C | 1.0105 C | -0.9175 C | 3030 |
| ice-air floor | 1.1126 C | 1.0096 C | -0.9121 C | 0 |

The 365d run passes and improves global A2 by `3.15%`, with small regional and
near-wall gains. Promote to a reproducibility repeat before replacing the GM0
candidate.

## Decision rules

- Reject if the annual repeat does not reproduce the climate gain.
- If repeated, promote the ice floor to the 0.45-degree production-like
  candidate and make GM0 the fallback.
- Do not couple to another closure in this first test.
