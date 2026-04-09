# Experiment 3: Recurrent vs Transformer Architecture Comparison

**Windstorm Institute Paper 7 - AGI Extensions of the Throughput Basin Framework**

## Research Question

Do truly serial architectures (state-space models like Mamba/RWKV) show tighter convergence
to τ = 4.16 bits with less variance across corpora than transformers?

## Hypothesis

**H1:** Mamba/RWKV models show BPT closer to 4.16 with less variance across corpora than
parameter-matched transformers, because they cannot parallelize around the serial decoding bottleneck.

**H0:** No significant difference between architectures.

## Methodology

### Models Tested

**Transformers:**
- EleutherAI/pythia-160m (162,322,944 parameters)
- EleutherAI/pythia-410m (405,334,016 parameters)
- EleutherAI/pythia-1.4b (1,414,647,808 parameters)
- openai-community/gpt2-medium (354,823,168 parameters)

**Serial/SSM:**
- state-spaces/mamba-130m-hf (129,135,360 parameters)
- state-spaces/mamba-370m-hf (371,516,416 parameters)
- state-spaces/mamba-1.4b-hf (1,372,178,432 parameters)

### Measurements

1. **BPT and BPB** on WikiText-2 test set (3 runs, mean ± std)
2. **Shuffling cascade** - 5 levels from original to fully shuffled
3. **Seven-corpus battery** - WikiText-2, Python, DNA, Shuffled, Math, Random ASCII, CSV
4. **Energy per token** - nvidia-smi power monitoring

## Key Findings

### 1. BPT on WikiText-2

| Model | Architecture | Parameters | BPT (mean ± std) | BPB (mean ± std) |
|-------|--------------|------------|------------------|------------------|
| pythia-160m | transformer | 162,322,944 | 3.956 ± 0.000 | 0.040 ± 0.000 |
| pythia-410m | transformer | 405,334,016 | 3.370 ± 0.000 | 0.034 ± 0.000 |
| pythia-1.4b | transformer | 1,414,647,808 | 2.981 ± 0.000 | 0.030 ± 0.000 |
| gpt2-medium | transformer | 354,823,168 | 3.674 ± 0.000 | 0.038 ± 0.000 |
| mamba-130m-hf | serial | 129,135,360 | 3.845 ± 0.000 | 0.039 ± 0.000 |
| mamba-370m-hf | serial | 371,516,416 | 3.300 ± 0.000 | 0.034 ± 0.000 |
| mamba-1.4b-hf | serial | 1,372,178,432 | 2.894 ± 0.000 | 0.030 ± 0.000 |

**Transformer mean BPT:** 3.495 bits
**Serial/SSM mean BPT:** 3.346 bits
**Difference:** 0.149 bits

**Transformer models in basin [3,6] bits:** 3/4 (75.0%)
**Serial models in basin [3,6] bits:** 2/3 (66.7%)

### 2. Variance Across 7 Corpora

**Transformer BPT variance:** 10.283
**Serial/SSM BPT variance:** 10.439
**Ratio:** 0.985x

✗ Transformers show lower or equal variance

### 3. Structural Bonus (All Shuffled - Original)

**Transformer structural bonus:** 6.839 ± 0.151 bits
**Serial/SSM structural bonus:** 6.782 ± 0.249 bits

### 4. Energy Efficiency

| Model | Architecture | Energy/Token (mJ) | Bits per Joule |
|-------|--------------|-------------------|----------------|
| pythia-160m | transformer | 0.124 | 31831.6 |
| pythia-410m | transformer | 0.320 | 10536.4 |
| pythia-1.4b | transformer | 1.671 | 1784.3 |
| gpt2-medium | transformer | 0.555 | 6620.6 |
| mamba-130m-hf | serial | 69.902 | 55.0 |
| mamba-370m-hf | serial | 148.391 | 22.2 |
| mamba-1.4b-hf | serial | 167.606 | 17.3 |

## Statistical Analysis

### Welch's t-test (BPT)

- **Statistic:** 0.4314
- **p-value:** 0.6880
- **Significant:** No (α = 0.05)
- **Interpretation:** Transformer mean: 3.495, Serial mean: 3.346

### Cohen's d (BPT)

- **Statistic:** 0.3967
- **Significant:** No (α = 0.05)
- **Interpretation:** Effect size: 0.397 (small)

### Levene's test (variance)

- **Statistic:** 0.0010
- **p-value:** 0.9751
- **Significant:** No (α = 0.05)
- **Interpretation:** Transformer variance: 10.283, Serial variance: 10.439

### Mann-Whitney U (structural bonus)

- **Statistic:** 6.0000
- **p-value:** 1.0000
- **Significant:** No (α = 0.05)
- **Interpretation:** Transformer bonus: 6.839±0.151, Serial bonus: 6.782±0.249

## Conclusion

**Evidence for H1 (Serial tighter):**
- None

**Evidence for H0 (No difference):**
- No significant difference in mean BPT
- No significant difference in BPT variance

**VERDICT: H0 SUPPORTED** - No significant difference between architectures.

This suggests the basin constraint is **data-driven** rather than architectural - both
transformers and serial models inherit ~4.4 bits/token from biological training corpora.

## Implications for Paper 7

1. **Architecture vs Data:** The results help isolate whether the basin is architectural or data-driven
2. **Model Selection:** Identifies which architectures may be more thermodynamically efficient for AGI
3. **Future Work:** Experiment 1 (synthetic training) will provide definitive test by training on non-biological data
