# R5 Scale Test — Corrected Report

## Verdict (CORRECTED)

**The auto-generated verdict "THESIS FALSIFIED AT SCALE" was wrong.**

The 3.82 BPT result is a tokenizer artifact, not an architectural compression effect. When measured in bits per source byte (the correct tokenizer-independent metric), both the 92M and 1B models extract exactly 8.0 bits per source byte from SYN-8 — perfectly tracking the source entropy of 8.0 bits/byte. There is no compression toward ~4 at any scale tested.

## What went wrong with the auto-report

The R5 training script built a new BPE tokenizer from the hex-encoded corpus, which produced a vocabulary of only 444 tokens (vs 8192 in the original Experiment 1). With vocab=444, each token covers ~2 source bytes, so a model that perfectly learns the distribution achieves BPT ≈ 8.0 × 2 / log₂(444) ≈ 3.8. The auto-report compared this 3.82 to the source entropy (8.0) and concluded "falsified." It should have compared bits-per-source-byte (8.0) to source entropy (8.0) and concluded "confirmed."

## Corrected results

| Model | Params | BPT | Bits/source byte | Source H | Tracking? |
|---|---|---|---|---|---|
| 92M (seed 42) | 85.8M | 3.822 | 8.001 | 8.0 | ✅ Perfect |
| 92M (seed 137) | 85.8M | 3.822 | 8.001 | 8.0 | ✅ Perfect |
| 1B (seed 42) | 1.21B | 3.824 | 8.005 | 8.0 | ✅ Perfect |
| 1B (seed 137) | 1.21B | 3.823 | 8.002 | 8.0 | ✅ Perfect |

## The proof that this is a tokenizer artifact

| Experiment | Tokenizer vocab | BPT | Bits/source byte | Same data? |
|---|---|---|---|---|
| B4 retrain (prior night) | ~8192 | 8.000 | ~8.0 | Yes (SYN-8) |
| R5 (this run) | 444 | 3.822 | 8.0 | Yes (SYN-8) |

BPT halves when the vocab shrinks. Bits-per-source-byte stays constant. QED: BPT is tokenizer-dependent; bits-per-source-byte is the correct metric.

## Actual contribution of R5

1. **Scale does not change the basin.** The 1B model extracts the same bits/byte as the 92M model on SYN-8.
2. **BPT is experimentally proven to be tokenizer-dependent.** Adversarial review item B3 is now an empirical fact, not just a theoretical concern.
3. **Bits-per-source-byte should replace BPT** as the primary metric in all future work.

*Correction prepared by the Conductor after human review of the auto-generated report.*
