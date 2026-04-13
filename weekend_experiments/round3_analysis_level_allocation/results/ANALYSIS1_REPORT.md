# Analysis 1: Level Allocation at INT4

## Key finding: The cliff is about LEVEL ALLOCATION, not just bit count

| Method | INT4 cosine | INT3 cosine | Cliff |
|---|---|---|---|
| symmetric_uniform | 0.905113 | 0.637253 | 0.2679 |
| NF4_normal_quantiles | 0.973153 | 0.947906 | 0.0252 |
| lloyd_max_optimal | 0.989730 | 0.965275 | 0.0245 |
| log_scale | 0.964533 | — | — |
| random_levels | 0.893958 | — | — |

## Interpretation

NF4 (0.9732) dramatically outperforms symmetric (0.9051) at INT4.
The cliff is not about 4 bits — it's about WHERE those 4 bits are placed.
Hardware implication: support non-uniform quantization tables, not just integer arithmetic.
