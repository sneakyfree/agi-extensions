# Analysis 3: Cross-Modal Structural Bonus

| Modality | Bonus (per unit) | Unit | Bonus × rate (bits/sec) |
|---|---|---|---|
| Language (text) | 6.70 | bits/token | 139 |
| Vision (spatial) | 0.69 | bits/pixel | 572314 |
| Audio (speech temporal) | 0.63 | bits/mel_dim | 3468 |
| Audio (music temporal) | 2.83 | bits/mel_dim | 15576 |
| PCFG-8 (synthetic hierarchy) | 5.33 | bits/token | 111 |

Language has the highest per-unit bonus (6.7 bits/token) but vision has the highest
total information rate because of the enormous pixel count per second.
