# Second industrial-comparison gate

Date: 2026-09-28  
Status: active gate

## Purpose

Prevent the project from becoming an internal-only tuning loop while the MOM6
annual run completes.  The second structured industrial target is NEMO.

## Rule

1. After the MOM6 Stage-F 365d v12 gate is scored, do **not** launch another
   solver-physics candidate before the NEMO Stage-F 30d smoke has either:
   - produced a comparable benchmark on the shared slice, or
   - been explicitly marked blocked with a recorded environment/source reason.
2. A 30d NEMO smoke is a protocol gate, not a climate claim.
3. The first annual NEMO comparison remains blocked until the 30d smoke passes
   the same stability and contract checks.

## First NEMO target

- model source: NEMO branch_5.0
- run: Stage-F 30d smoke;
- grid: shared 720x260 0.5-degree slice;
- bathymetry/init/reference: same MOM6-derived shared artifacts used by ocean_solver;
- forcing: Stage-F monthly wind plus 2m-air live-SST bulk proxy;
- sea ice: none;
- scoring: shared wet mask, final-10d surface RMSE/A2, stability gate;
- output: only metrics, manifest, provenance, and comparison table enter the repo.

## Current status

- NEMO source preparation has started on the external workspace.
- The official forge archive is downloading, but the link is slow and the first
  clone attempt disconnected.
- This is preparation, not a completed comparison.
