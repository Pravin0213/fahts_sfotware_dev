# run_analysis (3-D pipeline) vs OpenFOAM solidFoam — single free pipe

| case | level | cells | exposed end-cap faces | ΔT rise [K] | max \|Δ\| [K] | max \|Δ\|/rise | Δ mean final [K] | Tmax FAHTS / OF [°C] |
|---|---|---|---|---|---|---|---|---|
| engulfed | 0 | 456 | 48 | 966 | 21.4 | 2.22% | 2.32 | 986.8 / 986.4 |
| engulfed | 1 | 3648 | 192 | 967 | 8.7 | 0.90% | 0.045 | 986.8 / 986.8 |
| engulfed | 2 | 29184 | 768 | 967 | 5.82 | 0.60% | 0.0308 | 986.9 / 986.9 |
| outside | 0 | 456 | 48 | 932 | 41.4 | 4.45% | 6.87 | 916.5 / 951.7 |
| outside | 1 | 3648 | 192 | 946 | 16.4 | 1.73% | 1.71 | 958.7 / 965.8 |
| outside | 2 | 29184 | 768 | 950 | 11.4 | 1.20% | 0.295 | 968.5 / 970.3 |
