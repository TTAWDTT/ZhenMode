# Candidate Vertical-Mixing Regional Audit

## Global change

- Raw RMSE: `1.665 C` -> `1.625 C`.
- Global SSE improvement: `4.74%`.
- Mean warming: `0.056 C`.
- Cells warming: `81.1%`.

## Where the improvement lives

| region | SSE improvement share | delta RMSE |
|---|---:|---:|
| coast | 6.46% | -0.067 C |
| open deep | 4.87% | -0.039 C |
| near wall | 0.13% | -0.001 C |
| all | 4.74% | -0.040 C |

The coast improves more per cell than the deep ocean, but the deep
ocean still contributes more total SSE because of area.

## Stratification response

| quintile | old bias | new bias | SSE improvement share |
|---|---:|---:|---:|
| q1 | -0.552 | -0.518 | 3.20% |
| q2 | -1.131 | -1.081 | 4.81% |
| q3 | -1.322 | -1.288 | 3.13% |
| q4 | -1.465 | -1.402 | 5.53% |
| q5 | -1.583 | -1.484 | 5.98% |

The strongest-stratification quintile has the largest relative
improvement, but the absolute gain is still modest.

## Remaining worst errors

The 100 largest candidate errors are overwhelmingly in the
`300..360E` and `40..60N` sectors: 83 of 100 lie in the longitude
sector and 85 in the northern mid-latitude band. These are warm biases,
not the global cold bias. They are a separate problem from the broad
cold SST bias.

## Conclusion

Reduced vertical mixing is a useful candidate baseline, but its
improvement is modest and broadly distributed. It is not yet enough to
justify a full mixed-layer closure on its own. The next step should be
a focused audit of the high-latitude North Atlantic and coastal warm
biases, followed by a bulk heat-flux or boundary/ice diagnostic if
those regions remain dominant.
