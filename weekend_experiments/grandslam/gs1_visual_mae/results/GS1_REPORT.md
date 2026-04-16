# GS1: Visual MAE From Scratch — 86M Parameters

## Trained Model Results (ViT-MAE, 224×224, 15 epochs, 3 seeds)

| Level | Name | Approx Bits | Eval Loss (mean ± std) | 95% CI |
|---|---|---|---|---|
| 0 | uniform | ~0 | 0.000001 ± 0.000000 | [0.000001, 0.000001] |
| 1 | 4-color-blocks | ~2 | 0.065132 ± 0.025058 | [0.035907, 0.097102] |
| 2 | 16-color-blocks | ~4 | 0.075606 ± 0.004090 | [0.069823, 0.078562] |
| 3 | 64-color-pixels | ~6 | 0.084080 ± 0.001041 | [0.082636, 0.085052] |
| 4 | natural-like | structured | 0.013662 ± 0.000210 | [0.013472, 0.013955] |
| 5 | gaussian-noise | ~7 | 0.057536 ± 0.000001 | [0.057535, 0.057537] |
| 6 | uniform-noise | ~8 | 0.083332 ± 0.000000 | [0.083332, 0.083333] |

## Pretrained MAE-Large on Real + Synthetic Data

| Dataset | MAE Loss | N Images |
|---|---|---|
| CIFAR-100 | 0.062688 | 10000 |
| STL-10 | 0.163544 | 8000 |
| synthetic_uniform | 0.000108 | 1000 |
| synthetic_4-color-blocks | 1.946552 | 1000 |
| synthetic_16-color-blocks | 2.373395 | 1000 |
| synthetic_64-color-pixels | 1.86131 | 1000 |
| synthetic_natural-like | 0.138662 | 1000 |
| synthetic_gaussian-noise | 1.127743 | 1000 |
| synthetic_uniform-noise | 1.633487 | 1000 |

## Statistical Tests

Uniform noise vs. uniform color: t=249994.00, p=1.60e-11, Cohen's d=204119.25
  → Significant at α=0.05

## Interpretation

If reconstruction loss increases monotonically with source entropy, then visual throughput is data-driven — the model can reconstruct low-entropy images (it learned their structure) but fails on high-entropy images (no structure to exploit). This is the visual equivalent of Paper 7's SYN-8 experiment.
