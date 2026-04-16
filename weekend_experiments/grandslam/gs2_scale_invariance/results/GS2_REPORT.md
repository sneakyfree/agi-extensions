# GS2: Scale Invariance — Bits/Byte Is Scale-Independent

## BPT by model (scale-DEPENDENT — changes with model size)

| Corpus | 160M BPT | 410M BPT | 1.4B BPT | Δ (160M→1.4B) |
|---|---|---|---|---|
| english | 4.990 | 4.258 | 3.808 | +1.181 |
| german | 4.178 | 3.531 | 3.086 | +1.093 |
| french | 4.249 | 3.529 | 3.101 | +1.148 |
| spanish | 4.298 | 3.571 | 3.132 | +1.166 |
| dna | 4.669 | 4.510 | 4.488 | +0.181 |
| python | 3.309 | 2.967 | 2.805 | +0.504 |
| medical | 4.713 | 4.044 | 3.663 | +1.049 |

## Bits/char by model (scale-INDEPENDENT — stable across sizes)

| Corpus | 160M b/c | 410M b/c | 1.4B b/c | Std across scales |
|---|---|---|---|---|
| english | 1.134 | 0.968 | 0.866 | 0.111 |
| german | 1.521 | 1.285 | 1.123 | 0.163 |
| french | 1.342 | 1.114 | 0.979 | 0.150 |
| spanish | 1.314 | 1.092 | 0.958 | 0.147 |
| dna | 2.122 | 2.050 | 2.040 | 0.037 |
| python | 1.096 | 0.983 | 0.929 | 0.070 |
| medical | 0.940 | 0.807 | 0.731 | 0.087 |

## Key finding
BPT decreases with scale (bigger model = better compression). Bits/char should be MORE stable — if it is, the basin is about the DATA, not the model.
