# Stage-F 30d regional MLD probes

Same Stage-F dynamic-bulk contract and final-10d scoring window. These are
diagnostic runs only. The table separates a fixed 100m mixed-layer depth from
the stratification-derived MLD mode; they are different closures.

| run | ice | MLD treatment | verdict | global A2 | NA RMSE | near-wall RMSE | global bias | heat drift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Stage-F control | none | none | PASS | 1.692 C | 2.088 C | 1.958 C | -0.276 C | -0.184% |
| stratification MLD | dynamic | global stratification MLD | PASS | 1.173 C | 1.228 C | 0.941 C | -0.214 C | -0.078% |
| stratification MLD | none | 20--60N stratification MLD | PASS | 1.322 C | 0.827 C | 0.533 C | +0.063 C | -0.072% |
| constant MLD | none | 40--60N, fixed 100m | PASS | 1.562 C | 0.848 C | 0.533 C | -0.141 C | -0.143% |
| stratification MLD | none | 40--60N stratification MLD | PASS | 1.600 C | 1.246 C | 0.969 C | -0.195 C | -0.156% |
| combined Stage-I/MLD | dynamic | 40--60N, fixed 100m | PASS | 1.562 C | 0.848 C | 0.534 C | -0.141 C | -0.143% |

Interpretation: the fixed 100m 40--60N mixed layer is better than the
stratification-derived 40--60N closure on all three pre-registered SST metrics
in the 30d probe. Adding dynamic ice to this fixed-depth regional probe produces
only 22 ice cells in 30d and leaves all three metrics essentially unchanged. The
fixed-depth annual gate reversed the 30d regional gain and was not promoted.
