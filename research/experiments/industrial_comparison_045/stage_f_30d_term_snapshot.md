# Stage-F 30d term snapshot diagnostic

Source: final 30d Stage-F no-ice snapshot
`global_industrial_comparison_050_stage_f_30d_3d_terms/terms_00003.npy`.

Mean dT/dt by latitude band and physical term:

| term | -60..-40S | -40..-20S | -20..0 | 0..20N | 20..40N | 40..60N |
|---|---:|---:|---:|---:|---:|---:|
| advection | -9.6e-8 | -4.0e-8 | -3.0e-8 | +1.3e-8 | -3.0e-8 | +0.6e-9 |
| horizontal diffusion | +0.2e-9 | -0.2e-9 | -0.1e-9 | -0.0e-9 | -0.2e-9 | +0.4e-9 |
| vertical diffusion | -2.1e-9 | -4.4e-9 | -1.0e-9 | -0.1e-9 | +0.1e-9 | +0.2e-9 |
| convection | +7.8e-9 | +0.0e-9 | +5.5e-8 | +1.5e-7 | +5.2e-7 | +7.6e-7 |

Units are C/s. The largest diagnosed terms are convection in the tropics and
northern subtropics, with advection providing the main opposing cooling. This
supports an upper-ocean/ventilation diagnosis rather than horizontal-diffusion
tuning.
