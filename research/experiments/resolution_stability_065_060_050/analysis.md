# Finer-Resolution Stability and Skill Probe

Status: complete
Date: 2026-09-24
Baseline: `lambda80 + GM0 + min-depth500 + smooth80`, 0.50-degree candidate.

## Question

Can stabilized finer horizontal resolution improve the remaining North
Atlantic/near-wall error, and where do resolution gains stop paying for their
cost?

## Setup

All rungs use `dt=1800s`, `nu_h=2.0e6 m2/s`, and the 0.50-degree candidate
physics otherwise. Area remapping is used for 0.45 degree and finer. Every
365d run below passes the integration watchdog.

## 365d results

| resolution | remap | wall time | global A2 | NA 40--60N A2 | near-wall raw bias |
|---:|---|---:|---:|---:|---:|
| 0.70 | legacy | 11.9 min | 1.336 C | 1.025 C | -1.032 C |
| 0.65 | area | 21.7 min | 1.265 C | 0.952 C | not recorded |
| 0.60 | legacy | 22.4 min | 1.221 C | 0.940 C | not recorded |
| 0.55 | area | not run | — | — | — |
| 0.50 | legacy | 41.3 min | 1.171 C | 1.025 C | -0.923 C |
| 0.45 | area | 53.5 min | 1.149 C | 1.011 C | -0.917 C |
| 0.40 | area | 64.1 min | 1.142 C | 1.005 C | -0.937 C |
| 0.35 | area | 87.9 min | 1.114 C | 1.017 C | -0.965 C |

0.65/0.60 values use the lambda160 diagnostic candidate. 0.50--0.35 use the
locked lambda80 candidate setup.

## Decision

1. 0.45 degree is the best all-around production-like rung: it improves global
   and North Atlantic A2 without worsening the near-wall raw bias.
2. 0.40 gives a marginal global gain but worsens near-wall bias and costs more.
3. 0.35 improves global A2 only modestly, loses regional co-benefit, worsens
   near-wall bias, and costs about 88 min per 365d integration.
4. Stop the resolution scan here. If the 0.45-degree repeat matches the first
   run, promote it to the production-like candidate; otherwise retain 0.50.
5. The next physical lever is not more resolution or simple diffusion. Test
   boundary-current/lateral heat-transport closures.

## Coastal-closure side test

Enhanced coastal vertical diffusivity (`1e-5`, `1e-4`) and horizontal
diffusivity (`1e3`, `1e4`) did not improve the near-wall band. Reject simple
local diffusion as the missing closure; hard coastal restore remains a
diagnostic-only upper bound.
