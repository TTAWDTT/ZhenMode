# 三模式原计划设置摘要

本表列原计划 65 条的海面／温度误差摘要，保留失败。完整指标向量及单独追加组合见[数值与身份](data/configuration_scan_20261011.json)与[主报告](configuration_scan_results_zh.md)。

固定物理与原筛选；计时为各自硬件的 native worker 墙时，不能作 CPU／GPU 方法速度排名。

| 序号 | 算例 | 模式 | 设置 | 网格 | 步长(s) | 海面误差 | 温度RMS(°C) | 原筛选 | worker墙时(s) |
|---:|---|---|---|---|---:|---:|---:|---|---:|
| 0 | standing-wave | ZhenMode | base | 256×8×4 | 25.0 | 0.000026172136 | — | 通过 | 16.04 |
| 1 | standing-wave | ZhenMode | dt-half | 256×8×4 | 12.5 | 0.000026168884 | — | 通过 | 16.44 |
| 2 | standing-wave | ZhenMode | dt-quarter | 256×8×4 | 6.25 | 0.000026168741 | — | 通过 | 18.45 |
| 3 | standing-wave | ZhenMode | grid-half | 128×8×4 | 25.0 | 0.000026161103 | — | 通过 | 12.23 |
| 4 | standing-wave | ZhenMode | zhen-second-order | 256×8×4 | 25.0 | 0.00018008525 | — | 通过 | 11.83 |
| 5 | standing-wave | MOM6 | base | 256×8×4 | 25.0 | 0.0013616773 | — | 通过 | 9.23 |
| 6 | standing-wave | MOM6 | dt-half | 256×8×4 | 12.5 | 0.0018425020 | — | 通过 | 15.45 |
| 7 | standing-wave | MOM6 | dt-quarter | 256×8×4 | 6.25 | 0.00037416814 | — | 通过 | 28.89 |
| 8 | standing-wave | MOM6 | grid-half | 128×8×4 | 25.0 | 0.0014693109 | — | 通过 | 4.61 |
| 9 | standing-wave | MOM6 | mom-fast4 | 256×8×4 | 25.0 | 0.00044036729 | — | 通过 | 9.43 |
| 10 | standing-wave | MOM6 | mom-fast4-rk2b | 256×8×4 | 25.0 | 0.00044036729 | — | 通过 | 9.23 |
| 11 | standing-wave | MOM6 | mom-less-damping | 256×8×4 | 25.0 | 0.00042089985 | — | 通过 | 9.03 |
| 12 | standing-wave | Oceananigans | base | 256×8×4 | 25.0 | 0.00033259606 | — | 通过 | 123.98 |
| 13 | standing-wave | Oceananigans | dt-half | 256×8×4 | 12.5 | 0.00016067684 | — | 通过 | 132.60 |
| 14 | standing-wave | Oceananigans | dt-quarter | 256×8×4 | 6.25 | 0.000083937278 | — | 通过 | 178.12 |
| 15 | standing-wave | Oceananigans | grid-half | 128×8×4 | 25.0 | 0.00028658320 | — | 通过 | 104.52 |
| 16 | standing-wave | Oceananigans | oa-sub16 | 256×8×4 | 25.0 | 0.00024631509 | — | 通过 | 106.72 |
| 17 | standing-wave | Oceananigans | oa-sub32 | 256×8×4 | 25.0 | 0.00022503716 | — | 通过 | 114.75 |
| 18 | geostrophic-adjustment | ZhenMode | base | 8×256×4 | 50.0 | 0.000020391251 | 0 | 通过 | 11.23 |
| 19 | geostrophic-adjustment | ZhenMode | dt-half | 8×256×4 | 25.0 | 0.000020388742 | 0 | 通过 | 12.23 |
| 20 | geostrophic-adjustment | ZhenMode | dt-quarter | 8×256×4 | 12.5 | 0.000020388599 | 0 | 通过 | 14.44 |
| 21 | geostrophic-adjustment | ZhenMode | grid-half | 8×128×4 | 50.0 | 0.000020371144 | 0 | 通过 | 11.03 |
| 22 | geostrophic-adjustment | ZhenMode | zhen-second-order | 8×256×4 | 50.0 | 0.000031591443 | 0 | 通过 | 10.83 |
| 23 | geostrophic-adjustment | MOM6 | base | 8×256×4 | 50.0 | 0.0020269783 | 3.6866896E-14 | 通过 | 7.22 |
| 24 | geostrophic-adjustment | MOM6 | dt-half | 8×256×4 | 25.0 | 0.0010207813 | 4.9822122E-14 | 通过 | 12.44 |
| 25 | geostrophic-adjustment | MOM6 | dt-quarter | 8×256×4 | 12.5 | 0.0013683434 | 7.1683590E-14 | 通过 | 23.67 |
| 26 | geostrophic-adjustment | MOM6 | grid-half | 8×128×4 | 50.0 | 0.0020526535 | 3.6646228E-14 | 通过 | 4.01 |
| 27 | geostrophic-adjustment | MOM6 | mom-fast4 | 8×256×4 | 50.0 | 0.00057023658 | 3.6082510E-14 | 通过 | 7.42 |
| 28 | geostrophic-adjustment | MOM6 | mom-fast4-rk2b | 8×256×4 | 50.0 | 0.00057023658 | 3.6210071E-14 | 通过 | 7.42 |
| 29 | geostrophic-adjustment | MOM6 | mom-less-damping | 8×256×4 | 50.0 | 0.00055578935 | 3.7803074E-14 | 通过 | 7.42 |
| 30 | geostrophic-adjustment | Oceananigans | base | 8×256×4 | 50.0 | 0.00050432697 | 8.7738432E-15 | 通过 | 95.48 |
| 31 | geostrophic-adjustment | Oceananigans | dt-half | 8×256×4 | 25.0 | 0.00024828035 | 1.2377257E-14 | 通过 | 109.54 |
| 32 | geostrophic-adjustment | Oceananigans | dt-quarter | 8×256×4 | 12.5 | 0.00012123079 | 1.8217492E-14 | 通过 | 131.59 |
| 33 | geostrophic-adjustment | Oceananigans | grid-half | 8×128×4 | 50.0 | 0.00047612781 | 9.0404123E-15 | 通过 | 99.10 |
| 34 | geostrophic-adjustment | Oceananigans | oa-sub16 | 8×256×4 | 50.0 | 0.00032354133 | 9.0491475E-15 | 通过 | 95.49 |
| 35 | geostrophic-adjustment | Oceananigans | oa-sub32 | 8×256×4 | 50.0 | 0.00025896678 | 8.9150873E-15 | 通过 | 96.88 |
| 36 | thermal-wind | ZhenMode | base | 32×32×16 | 50.0 | 0.00066778363 | 0.000021419114 | 通过 | 13.43 |
| 37 | thermal-wind | ZhenMode | dt-half | 32×32×16 | 25.0 | 0.00066744647 | 0.000021419337 | 通过 | 16.04 |
| 38 | thermal-wind | ZhenMode | dt-quarter | 32×32×16 | 12.5 | 0.00066737742 | 0.000021419404 | 通过 | 21.66 |
| 39 | thermal-wind | ZhenMode | grid-half | 16×16×16 | 50.0 | 0.0026004409 | 0.000034816759 | 通过 | 13.64 |
| 40 | thermal-wind | ZhenMode | zhen-second-order | 32×32×16 | 50.0 | 0.0021163068 | 0.00013024872 | 通过 | 13.03 |
| 41 | thermal-wind | ZhenMode | grid-double | 64×64×16 | 50.0 | 0.00017069463 | 0.000010246069 | 通过 | 16.85 |
| 42 | thermal-wind | ZhenMode | vertical-double | 32×32×32 | 50.0 | 0.00066774598 | 0.000020668431 | 通过 | 21.46 |
| 43 | thermal-wind | ZhenMode | zhen-centered-tracer | 32×32×16 | 50.0 | 0.00066778344 | 0.000033378588 | 通过 | 14.04 |
| 44 | thermal-wind | MOM6 | base | 32×32×16 | 50.0 | 0.00081156625 | 0.00020441055 | 通过 | 21.87 |
| 45 | thermal-wind | MOM6 | dt-half | 32×32×16 | 25.0 | 0.00082788779 | 0.00020441550 | 通过 | 43.13 |
| 46 | thermal-wind | MOM6 | dt-quarter | 32×32×16 | 12.5 | 0.00084665885 | 0.00020432112 | 通过 | 90.27 |
| 47 | thermal-wind | MOM6 | grid-half | 16×16×16 | 50.0 | 0.0032426033 | 0.00080055654 | 通过 | 6.42 |
| 48 | thermal-wind | MOM6 | mom-fast4 | 32×32×16 | 50.0 | 0.00084031707 | 0.00020441053 | 通过 | 23.27 |
| 49 | thermal-wind | MOM6 | mom-fast4-rk2b | 32×32×16 | 50.0 | 0.00084031707 | 0.00020441053 | 通过 | 24.27 |
| 50 | thermal-wind | MOM6 | mom-less-damping | 32×32×16 | 50.0 | 0.00084578203 | 0.00020441054 | 通过 | 24.47 |
| 51 | thermal-wind | MOM6 | grid-double | 64×64×16 | 50.0 | — | — | RuntimeError: mom6 worker failed (1); see /mnt/d/Github/ocean-solver/outputs/configuration-scan-20261011/row-51-thermal-wind-mom6-grid-double/mom6/run.log | 145.18 |
| 52 | thermal-wind | MOM6 | vertical-double | 32×32×32 | 50.0 | 0.00081206808 | 0.00020554149 | 通过 | 39.10 |
| 53 | thermal-wind | MOM6 | mom-ppm-h3 | 32×32×16 | 50.0 | 0.00084031976 | 0.00020056717 | 通过 | 26.28 |
| 54 | thermal-wind | MOM6 | mom-ppm | 32×32×16 | 50.0 | 0.00084026514 | 0.00021115869 | 通过 | 25.47 |
| 55 | thermal-wind | Oceananigans | base | 32×32×16 | 50.0 | 0.017427674 | 0.00019171467 | 通过 | 118.96 |
| 56 | thermal-wind | Oceananigans | dt-half | 32×32×16 | 25.0 | 0.017573191 | 0.00019171486 | 通过 | 148.86 |
| 57 | thermal-wind | Oceananigans | dt-quarter | 32×32×16 | 12.5 | 0.017652743 | 0.00019171496 | 通过 | 209.47 |
| 58 | thermal-wind | Oceananigans | grid-half | 16×16×16 | 50.0 | 0.018549660 | 0.00075298234 | 通过 | 114.76 |
| 59 | thermal-wind | Oceananigans | oa-sub16 | 32×32×16 | 50.0 | 0.017428319 | 0.00019171466 | 通过 | 116.96 |
| 60 | thermal-wind | Oceananigans | oa-sub32 | 32×32×16 | 50.0 | 0.017437217 | 0.00019171467 | 通过 | 116.96 |
| 61 | thermal-wind | Oceananigans | grid-double | 64×64×16 | 50.0 | 0.017198122 | 0.000046150275 | 通过 | 120.37 |
| 62 | thermal-wind | Oceananigans | vertical-double | 32×32×32 | 50.0 | 0.0086117849 | 0.00019175305 | 通过 | 115.96 |
| 63 | thermal-wind | Oceananigans | oa-centered4 | 32×32×16 | 50.0 | 0.017428318 | 0.00019097526 | 通过 | 116.16 |
| 64 | thermal-wind | Oceananigans | oa-weno5 | 32×32×16 | 50.0 | 0.017428309 | 0.00019090986 | 通过 | 137.63 |
