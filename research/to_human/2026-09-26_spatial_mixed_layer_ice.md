# Spatially restricted mixed-layer/ice closure

Date: 2026-09-26  
Status: validated diagnostic at 0.5 degree

## Key result

A 20m mixed-layer/ice closure restricted to 55--65N passes the annual gate:

| metric | 0.5° ice-floor control | 55--65N mixed-layer/ice |
|---|---:|---:|
| global A2 RMSE | 1.128 C | 1.127 C |
| NA 40--60N raw RMSE | 0.924 C | 0.872 C |
| near-wall raw bias | -0.921 C | -0.604 C |
| near-wall raw RMSE | 1.085 C | 0.888 C |

The annual repeat reproduces to numerical precision.

## Why it matters

The earlier uniform mixed-layer closure improved the near-wall/polar bias but
damaged the subtropics.  Restricting the closure to the high-latitude band
keeps the low-latitudes unchanged and still improves the North Atlantic.

## Honest caveat

This is still a minimal closure, not CICE/SI3.  It adds:
- mixed-layer heat capacity,
- freezing-point salt flux,
- ice-air floor,
- a latitude-masked mixed-layer depth.

It should be called a **diagnostic sea-ice/mixed-layer closure**, not a
production sea-ice model.
