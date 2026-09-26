# Stage-F 30d MLD band probes

Same Stage-F dynamic-bulk contract and final-10d scoring window. The rows below
are diagnostic only. The 40--60N run named `constant_mld` uses a fixed 100 m
mixed-layer depth restricted to that latitude band; it is **not** the
stratification-derived MLD mode. Keep this distinction explicit to avoid
overstating the closure test.

| run | ice | MLD treatment | verdict | global A2 | NA RMSE | near-wall RMSE | global bias | heat drift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Stage-F control | none | none | PASS | 1.692 C | 2.088 C | 1.958 C | -0.276 C | -0.184% |
| stratification MLD | dynamic | global stratification MLD | PASS | 1.173 C | 1.228 C | 0.941 C | -0.214 C | -0.078% |
| stratification MLD | none | 20--60N stratification MLD | PASS | 1.322 C | 0.827 C | 0.533 C | +0.063 C | -0.072% |
| constant MLD | none | 40--60N, fixed 100 m | PASS | 1.562 C | 0.848 C | 0.533 C | -0.141 C | -0.143% |

Interpretation: restricting a fixed 100 m mixed layer to 40--60N strongly
improves North Atlantic and near-wall SST errors, while global A2 remains worse
than the dynamic-ice stratification probe. This is useful attribution, not a
promoted closure. The correct next annual test is the 40--60N
stratification-derived MLD mode, not the constant-depth run.
