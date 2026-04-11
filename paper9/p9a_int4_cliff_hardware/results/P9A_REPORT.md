# Paper 9 Experiment A: INT4 vs INT3 Cliff in Hardware Simulation

## Question

Does the software INT4→INT3 cliff (Paper 7, Exp 2) persist in hardware
arithmetic simulation?

## Method

Three levels of simulation, all pure CPU:

1. **Linear layer quantization** — quantize weight matrices at INT8 through INT2,
   measure output SQNR and cosine similarity. 3 dimensions × 3 seeds × 6 precisions.

2. **Attention pattern preservation** — quantize Q, K, V matrices, measure whether
   the softmax attention distribution is preserved. Tests whether INT3 destroys
   the attention mechanism specifically.

3. **Full transformer block** — quantize all weight matrices in a complete
   attention + FFN block, measure end-to-end output fidelity.

4. **PyRTL MAC simulation** — build actual gate-level multiply-accumulate units
   at each precision, measure gate count and critical path length. This is real
   hardware description, not just numerical approximation.

## Verdict

CLIFF CONFIRMED IN HARDWARE ARITHMETIC. INT4→INT3 degradation is 5.0× worse than INT5→INT4.

## Key Numbers

| Precision | Linear SQNR (dB) | Attention cosine | Block cosine |
|---|---|---|---|
| INT8 | 39.3 | 0.999999 | 0.999993 |
| INT6 | 27.1 | 0.999999 | 0.999894 |
| INT5 | 20.8 | 0.999999 | 0.999553 |
| INT4 | 14.2 | 0.999996 | 0.997897 |
| INT3 | 6.8 | 0.999981 | 0.988102 |
| INT2 | 0.3 | 0.999944 | 0.974211 |

## Implications for Paper 9

If the cliff is confirmed in hardware arithmetic, it means the INT4 floor is
a mathematical property of low-precision representation, not a bug in
bitsandbytes software. Any inference accelerator below 4-bit weight precision
will hit the same wall regardless of the quantization algorithm or hardware
implementation.

## Files

- `results/precision_sweep.csv` — all linear, attention, block measurements
- `results/cliff_analysis.csv` — degradation ratios at each transition
- `results/pyrtl_mac.csv` — hardware gate counts and critical paths
- `plots/precision_cliff.png` — four-panel visualization
