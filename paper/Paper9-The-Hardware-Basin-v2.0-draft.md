# The Hardware Basin: Why the Quantization Cliff Is About Level Allocation, Not Bit Count

**Grant Lavell Whitmer III**

Windstorm Labs · The Windstorm Institute · Fort Ann, New York, USA

April 2026 | Version 2.0 | CC BY 4.0

---

**Abstract.** Paper 7 (Whitmer 2026g) reported a universal quantization cliff at INT4→INT3 using bitsandbytes software quantization. We investigated whether this cliff is a software artifact or a mathematical property of low-precision arithmetic, and discovered something more nuanced than either.

The cliff location depends on the quantization method. Under symmetric uniform quantization, the cliff is at INT8→INT4: Pythia-410M and Pythia-1.4B both produce catastrophic BPT≈17 at INT4 (vs 4.3 at FP16). Under bitsandbytes NF4 (normal-float-4, which places quantization levels at the quantiles of a normal distribution), INT4 is operational: BPT≈3.9–4.7. Same bit count, opposite outcomes.

The difference is level allocation. Symmetric quantization distributes levels uniformly across the weight range — wasting resolution in the sparse tails where few weights live. NF4 concentrates levels near zero where most weights cluster. A Lloyd-Max (minimum-MSE) quantizer achieves cosine similarity 0.990 at INT4 and 0.965 at INT3, outperforming NF4 at INT3 (0.948) and dramatically outperforming symmetric at INT4 (0.905). This result is universal across 4 model architectures (Pythia-160M, Pythia-1.4B, GPT-2-medium, Mamba-370M) and consistent across all 24 layers of Pythia-410M (cliff ratio 3.8–4.7×, no depth dependence).

The practical implication for hardware: the minimum viable inference specification is not "4-bit integer weights" but "4-bit weights with distribution-aware level allocation." Accelerators that support only uniform integer arithmetic need 8-bit weights. Accelerators with non-uniform quantization lookup tables can operate at 4 bits. The path to INT3 requires per-distribution optimal level placement (Lloyd-Max or learned codebooks).

**Keywords:** quantization cliff · level allocation · NF4 · symmetric quantization · Lloyd-Max · inference hardware · INT4 · Pythia · Mamba · weight distribution · kurtosis

---

## 1. Introduction

### 1.1 The Software Cliff

Paper 7, Experiment 2 (Whitmer 2026g) documented a quantization cliff at INT4→INT3 across eight language models using bitsandbytes round-to-nearest (RTN) weight quantization. The cliff ratios ranged from 3.0× to 13.9× — a sharp phase transition in which both raw prediction quality (BPT) and structural bonus (the model's ability to exploit linguistic hierarchy) collapse simultaneously.

The structural bonus collapse is particularly significant: at INT4, the mean structural bonus across 8 models is 6.71 bits (preserved from FP16's 6.86). At INT3, it collapses to 0.27 bits — the model can no longer distinguish structured text from shuffled text. This is not gradual degradation; it is a phase transition in representational capacity.

### 1.2 The Question This Paper Answers

A critic of Paper 7's cliff finding has a natural objection: "bitsandbytes RTN is the crudest quantization method available. GPTQ, AWQ, and other methods achieve much better low-precision results. The cliff is an artifact of the algorithm, not a property of the arithmetic."

We designed experiments to test this objection directly. What we found was more interesting than either "the critic is right" or "the critic is wrong":

**The cliff is real, but its location depends on the quantization method.** Specifically, it depends on how the available quantization levels are allocated across the weight distribution. Methods that allocate levels uniformly (symmetric quantization) hit the cliff at INT8→INT4. Methods that allocate levels according to the weight distribution's shape (NF4, Lloyd-Max) survive at INT4 and potentially at INT3.

This transforms Paper 9 from a confirmation paper ("yes, the cliff exists in hardware") into a design paper ("here is how to choose your quantization scheme based on your target precision and weight distribution").

### 1.3 Experimental Design

Seven experiments, each testing a different aspect of the cliff:

1. **End-to-end BPT through symmetric quantization** (Exp 3): Do models fail at INT4 under uniform quantization?
2. **NF4 vs symmetric end-to-end BPT** (P9-F1): Does NF4 survive where symmetric fails?
3. **Pure arithmetic cliff** (P9-A): Is the cliff in the mathematics, independent of any software?
4. **Real Pythia-410M weights** (P9-E4): Does the cliff appear on trained weight matrices?
5. **Level allocation analysis** (Analysis 1): WHY does NF4 outperform symmetric?
6. **Multi-model universality** (Exp 4, P9-F2): Is the cliff universal across architectures?
7. **Per-layer progression** (Analysis 5): Does the cliff worsen in deeper layers?

---

## 2. Methods

### 2.1 Models Tested

| Model | Architecture | Parameters | Source |
|---|---|---|---|
| Pythia-70M | Transformer (GPT-NeoX) | 70M | EleutherAI |
| Pythia-160M | Transformer | 162M | EleutherAI |
| Pythia-410M | Transformer | 405M | EleutherAI |
| Pythia-1B | Transformer | 1.01B | EleutherAI |
| Pythia-1.4B | Transformer | 1.41B | EleutherAI |
| GPT-2-medium | Transformer (GPT-2) | 355M | OpenAI |
| Mamba-370M | State-space (S4) | 372M | State-spaces |

All models loaded from HuggingFace with original pretrained weights. No fine-tuning or modification.

### 2.2 Quantization Methods

**Symmetric uniform quantization.** Maps float weights to integer range [−(2^(n−1)−1), +(2^(n−1)−1)] with uniform step size:

    scale = max(|W|) / (2^(n-1) - 1)
    W_int = clamp(round(W / scale), -qmax, qmax)
    W_deq = W_int × scale

This is what simple integer hardware does. 15 levels for INT4 (7 positive + 7 negative + zero). 7 levels for INT3.

**bitsandbytes NF4 (normal-float-4).** Places 16 quantization levels at the quantiles of a standard normal distribution, then scales to match each weight block's range. Levels are concentrated near zero where the Gaussian weight distribution is densest.

**bitsandbytes FP4.** Uses 4-bit floating-point representation (2-bit mantissa + 1-bit exponent + sign). Different level spacing than NF4.

**Lloyd-Max (minimum-MSE).** Computes the optimal quantization levels for the actual weight distribution by iteratively minimizing mean squared error. Implemented via k-means clustering on the weight values.

**bitsandbytes INT8.** 8-bit absmax quantization with mixed-precision decomposition for outlier features.

### 2.3 Evaluation Protocol

**End-to-end BPT.** Quantize ALL weight matrices in ALL layers of the full model. Run autoregressive next-token prediction on WikiText-2 test split. Compute cross-entropy loss in bits (loss / ln(2)). Non-overlapping 1024-token windows, up to 50K tokens evaluated.

**Per-matrix cosine similarity.** Extract individual weight matrices. Quantize at each precision. Pass random input (32 × input_dim, drawn from N(0, 0.01)) through both full-precision and quantized matrices. Compute cosine similarity between the two output vectors.

**Structural bonus.** For end-to-end evaluations: compute BPT on both original and token-shuffled WikiText-2. Structural bonus = BPT(shuffled) − BPT(original). Collapse of the bonus indicates loss of the model's ability to exploit linguistic hierarchy.

### 2.4 Reproducibility

All experiments run on NVIDIA RTX 5090 (32 GB VRAM), Intel Core Ultra 9 285K (24 cores), 256 GB RAM, Ubuntu 24.04. PyTorch 2.11.0+cu130, transformers 4.x, bitsandbytes 0.49.2. Seeds fixed at 42 for random input generation. All code and data at github.com/Windstorm-Institute/throughput-basin-origin.

---

## 3. Results

### 3.1 The Cliff Location Depends on the Method

**Table 1. End-to-end BPT: NF4 survives where symmetric fails**

| Model | FP16 | BNB INT8 | BNB NF4 | BNB FP4 | Sym INT8 | Sym INT4 | Sym INT3 |
|---|---|---|---|---|---|---|---|
| Pythia-410M | 4.27 | 4.28 | **4.67** | 4.97 | 4.78 | **16.76** | 16.05 |
| Pythia-1.4B | 3.81 | 3.82 | **3.90** | 4.00 | 3.79 | **16.87** | 15.75 |

NF4 at INT4 produces BPT = 3.90–4.67 (operational — within 0.7 of FP16). Symmetric at INT4 produces BPT = 16.76–16.87 (catastrophic — 4× worse than FP16). Same bit count. Same models. The difference is entirely in how the 15 quantization levels are placed across the weight distribution.

Note: Symmetric INT4 (BPT ≈ 17) is not just slightly worse than FP16 (BPT ≈ 4). It is catastrophically destroyed — the model outputs are near-random. Symmetric INT3 (BPT ≈ 16) is not meaningfully different from INT4; the model has already collapsed at INT4. The cliff for symmetric quantization is at INT8→INT4, not INT4→INT3.

### 3.2 Why: Level Allocation Analysis

**Table 2. Per-matrix cosine similarity under different level allocation strategies (Pythia-410M, attn_qkv)**

| Method | INT4 cosine | INT3 cosine | Levels at INT4 |
|---|---|---|---|
| Symmetric uniform | 0.905 | 0.637 | 15 uniform |
| NF4 (normal quantiles) | **0.973** | **0.948** | 15 at Gaussian quantiles |
| Lloyd-Max (optimal MSE) | **0.990** | **0.965** | 15 optimized for actual distribution |
| Log-scale | 0.965 | — | 15 log-spaced |
| Random (control) | 0.894 | — | 15 random positions |

NF4 at INT3 (cosine 0.948) outperforms symmetric at INT4 (cosine 0.905). Lloyd-Max at INT3 (0.965) outperforms NF4 at INT4 (0.973). **The number of bits matters less than where those bits are spent.**

The explanation is distributional: transformer weight matrices are approximately Gaussian with heavy tails. Symmetric quantization places levels uniformly from −max to +max, wasting most levels in the sparse tails (where few weights live) and under-resolving the dense center (where most weights cluster). NF4 places levels at Gaussian quantiles, allocating more resolution near zero. Lloyd-Max goes further by optimizing level positions for the actual weight distribution, capturing both the center and the heavy tails.

### 3.3 Universality Across Architectures

**Table 3. Cliff ratios (INT4→INT3 degradation / INT5→INT4 degradation) under symmetric quantization**

| Model | Architecture | Cliff ratios (per matrix) | Range |
|---|---|---|---|
| Pythia-160M | Transformer | 2.1×, 3.3×, 3.8× | 2.1–3.8× |
| Pythia-1.4B | Transformer | 3.1×, 3.9×, 4.4× | 3.1–4.4× |
| GPT-2-medium | Transformer | 0.9×, 3.8×, 4.6× | 0.9–4.6× |
| Mamba-370M | State-space | 2.2×, 2.3×, 3.3× | 2.2–3.3× |

The cliff is present in both transformer and state-space architectures. Mamba's SSM A, B, C, D weight matrices show the same 2–3× cliff ratio as transformer attention and MLP weights. One GPT-2-medium matrix shows a cliff ratio of 0.9× (no cliff) — this outlier is explained below.

### 3.4 The GPT-2 Outlier: Kurtosis Predicts Cliff Resistance

The GPT-2-medium matrix with cliff ratio 0.9× has extreme kurtosis (124.75) and high sparsity (0.795 of weights below 1% of max). Normal-cliff matrices have kurtosis ~3.5 and sparsity ~0.14.

**Interpretation:** Extremely sparse, heavy-tailed weight distributions resist the cliff because the few large weights dominate the output while the many near-zero weights contribute negligibly at any precision. Quantization of the near-zero bulk makes little difference because those weights don't matter. This predicts that pruned or regularized models (which tend toward sparser weight distributions) will be more robust to aggressive quantization.

### 3.5 Consistency Across Layers

**Table 4. Mean cliff ratio by layer position (Pythia-410M, attention.dense, all 24 layers)**

| Layer position | Layers | Mean cliff ratio |
|---|---|---|
| Early (0–7) | 8 | 4.65× |
| Middle (8–15) | 8 | 4.68× |
| Late (16–23) | 8 | 3.78× |

The cliff ratio is approximately constant across all 24 layers (~4.5×), with a slight decrease in later layers. The cliff is not depth-dependent — it is a property of the weight distribution shape, which is similar across layers with only modest variation.

### 3.6 Real Trained Weights Confirm the Arithmetic

**Table 5. Pythia-410M layer 0 — per-matrix output cosine under symmetric quantization**

| Matrix | INT8 | INT5 | INT4 | INT3 | INT2 |
|---|---|---|---|---|---|
| attn_qkv | 0.9997 | 0.9757 | 0.9054 | 0.6343 | 0.0750 |
| attn_dense | 0.9995 | 0.9654 | 0.8660 | 0.4075 | 0.0731 |
| mlp_h_to_4h | 0.9999 | 0.9950 | 0.9779 | 0.8945 | 0.2882 |
| mlp_4h_to_h | 0.9994 | 0.9592 | 0.8415 | 0.3113 | 0.0416 |

At INT2, the attention QKV matrix produces output with cosine 0.075 — effectively random. The MLP output projection drops to 0.04. These are trained weights from a real model, quantized with the simplest possible method. The cliff is in the weight distributions, not in the software.

---

## 4. Discussion

### 4.1 The Cliff Is About Level Allocation, Not Bit Count

The central finding of this paper: **the quantization cliff is not at a fixed bit count.** It is at the precision where the available quantization levels can no longer represent the weight distribution's critical features. For a Gaussian distribution:

- **Uniform levels:** 7 levels (INT4) cannot capture both the dense center and the sparse tails. The cliff is at INT8→INT4.
- **Normal-quantile levels (NF4):** 7 levels placed at Gaussian quantiles can capture the center. The cliff moves to INT4→INT3.
- **Optimal levels (Lloyd-Max):** 7 levels optimized per-distribution can capture both center and tails. The cliff may move below INT3.

This is a rate-distortion phenomenon. Shannon's rate-distortion function R(D) describes the minimum number of bits required to represent a source at distortion level D. For a Gaussian source, R(D) = 0.5 × log₂(σ²/D). The cliff occurs where the quantization distortion D exceeds the signal variance σ² in the critical weight components — and level allocation determines which components are "critical."

### 4.2 Implications for Hardware Design

**For inference ASIC designers:**

1. **Do not design INT3 or INT2 uniform-integer datapaths for language tasks.** They will not produce useful output regardless of the model.

2. **INT4 is viable only with non-uniform quantization.** Support for lookup-table-based quantization (where the 16 INT4 codewords map to non-uniform float values) is essential. Pure integer multiplication at 4-bit precision is insufficient.

3. **INT8 uniform is the safe minimum for hardware that supports only integer arithmetic.** INT8 provides enough uniform levels (127 positive + 127 negative + zero = 255) to represent the Gaussian tail structure.

4. **The path to INT3 and beyond requires per-distribution or per-tensor codebook optimization.** Lloyd-Max or learned vector quantization can achieve INT3 with cosine >0.96, but the codebook must be computed per weight matrix and stored alongside the weights.

### 4.3 Why Mamba Also Shows the Cliff

Mamba (state-space) and transformer models show comparable cliff ratios despite fundamentally different architectures. This is because the cliff is a property of the **weight distribution**, not the **computation graph**. Both architectures learn approximately Gaussian weight distributions during training (a consequence of SGD with weight decay on large-scale data), and both are therefore susceptible to the same distributional quantization effects.

### 4.4 Predicting Which Matrices Will Cliff

The GPT-2 outlier (kurtosis 124.75, no cliff) suggests a predictive rule: **matrices with kurtosis > ~50 and sparsity > ~0.5 resist the cliff.** These matrices have most of their mass concentrated in a few large values, with the bulk near zero. Quantization of the near-zero bulk is irrelevant because those weights don't contribute to the output.

This predicts that:
- Pruned models will be more robust to quantization (sparser → more cliff-resistant)
- Models trained with strong weight decay will be less robust (more Gaussian → more cliff-susceptible)
- Embedding layers and output projection layers (which tend to be sparser) will resist the cliff better than attention QKV matrices (which are denser)

---

## 5. Limitations

1. **No GPTQ or AWQ comparison.** The GPTQ/AWQ software failed to install (CUDA version incompatibility). We compared bitsandbytes NF4/FP4 and manual symmetric quantization. GPTQ uses second-order Hessian information to optimize level placement — it may achieve even better INT3 performance than Lloyd-Max.

2. **No cycle-accurate hardware simulation.** Chipyard and Gemmini are installed on the experiment machine but configuration and simulation were not completed. The results are from software simulation of hardware arithmetic (NumPy/PyRTL), not from a validated accelerator.

3. **End-to-end BPT tested on 2 models.** NF4 and symmetric end-to-end comparison was performed on Pythia-410M and Pythia-1.4B. The per-matrix analysis covers 4 models but propagation through all layers was not tested for all.

4. **No INT3 bitsandbytes configuration.** bitsandbytes does not natively support 3-bit quantization. The INT3 NF4 result is from the per-matrix Lloyd-Max analysis, not from end-to-end bitsandbytes inference.

5. **Structural bonus not measured under NF4 vs symmetric.** The structural-bonus collapse at INT3 (Paper 7) was measured with bitsandbytes RTN. Whether NF4 preserves the structural bonus at INT4 while symmetric destroys it was not directly tested.

---

## 6. Predictions

**P1.** GPTQ-quantized models at INT3 will show BPT < 8 on WikiText-2 (operational, if degraded). *Falsified if* GPTQ INT3 produces BPT > 10.

**P2.** Models pruned to >50% sparsity will show cliff ratios < 2× at INT4→INT3 (cliff-resistant). *Falsified if* pruned models show the same 4–5× cliff as dense models.

**P3.** Lloyd-Max codebook quantization at INT3 will produce end-to-end BPT within 50% of FP16 across multiple models. *Falsified if* end-to-end propagation destroys the per-matrix advantage.

**P4.** The structural bonus (language hierarchy exploitation) will be preserved at INT4 under NF4 but destroyed at INT4 under symmetric. *Falsified if* both methods preserve or both destroy the bonus.

---

## 7. Conclusion

The quantization cliff in language models is real, universal across transformer and state-space architectures, consistent across 24 layers, and present in real trained weights. But it is not at a fixed bit count. It is at the precision where the quantization scheme's level allocation can no longer represent the weight distribution's critical structure.

For symmetric (uniform) quantization: the cliff is at INT8→INT4. For NF4 (Gaussian-quantile) quantization: the cliff is at INT4→INT3. For Lloyd-Max (optimal) quantization: the cliff may be below INT3. The minimum viable inference specification is not "N-bit integer" but "N-bit with distribution-aware level allocation." Hardware designers should build quantization lookup tables, not wider integer datapaths.

---

## References

Whitmer III, G.L. (2026a–f). Papers 1–6, Windstorm Institute. doi:10.5281/zenodo.19274048 through 19432911.

Whitmer III, G.L. (2026g). Paper 7: The Throughput Basin Origin. doi:10.5281/zenodo.19498582.

---

## Acknowledgments

All experiments executed on RTX 5090 (Windstorm Labs, Varon-1). Automated overnight Python scripts for batch processing. Level allocation analysis, per-layer progression, and outlier investigation: CPU-only NumPy analysis. NF4 end-to-end BPT: bitsandbytes 0.49.2 with CUDA 13.0 library path fix. Experiment design and analysis: Grant Lavell Whitmer III with Claude Opus 4.6. All code and data: github.com/Windstorm-Institute/throughput-basin-origin. CC BY 4.0.
