# Stage-F 30d stratification-MLD probes

Same Stage-F dynamic-bulk contract and final-10d scoring window. `no-ice` means
the probe removes the explicit sea-ice closure, so these are diagnostic runs and
not yet the preferred climate configuration.

| run | ice | MLD band | verdict | global A2 | NA RMSE | near-wall RMSE | global bias | heat drift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Stage-F control | none | none | PASS | 1.692 C | 2.088 C | 1.958 C | -0.276 C | -0.184% |
| stratification MLD | dynamic | global | PASS | 1.173 C | 1.228 C | 0.941 C | -0.214 C | -0.078% |
| stratification MLD | none | 20--60N | PASS | 1.322 C | 0.827 C | 0.533 C | +0.063 C | -0.072% |
| stratification MLD | none | 40--60N | PASS | 1.562 C | 0.848 C | 0.533 C | -0.141 C | -0.143% |

Interpretation: removing ice and limiting stratification MLD to 40--60N strongly
improves North Atlantic and near-wall SST errors, while global A2 remains worse
than the dynamic-ice stratification probe. The next annual gate is the 40--60N
no-ice variant; it is justified by the 30d regional/wall improvement but must
survive the 365d stability and error gates before promotion.
