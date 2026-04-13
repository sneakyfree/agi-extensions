# Analysis 2: GPT-2 Outlier Matrix

## Cliff ratios for GPT-2-medium

| Matrix | Cliff ratio | Std | Kurtosis | Sparsity |
|---|---|---|---|---|
| ...mer.h.0.mlp.c_proj.weight | 0.11× **OUTLIER** | 0.083322 | 46.21 | 0.446 |
| ...ormer.h.1.mlp.c_fc.weight | 0.40× **OUTLIER** | 0.105658 | 5.12 | 0.371 |
| ...mer.h.1.mlp.c_proj.weight | 0.89× **OUTLIER** | 0.079143 | 124.75 | 0.795 |
| ...ormer.h.0.mlp.c_fc.weight | 3.72× | 0.101999 | 3.47 | 0.147 |
| ...ormer.h.2.mlp.c_fc.weight | 3.83× | 0.106556 | 3.55 | 0.136 |
| ...mer.h.2.mlp.c_proj.weight | 4.58× | 0.081725 | 216.82 | 0.796 |

## Interpretation

Outlier kurtosis: 58.69, Normal kurtosis: 74.61
Outlier has LOWER kurtosis (lighter tails) → fewer extreme weights → less to lose at INT3.
