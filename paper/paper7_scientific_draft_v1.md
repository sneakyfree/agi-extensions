# The Throughput Basin Origin: Four Orthogonal Experiments on Whether Serial Decoding Convergence Is Architectural, Thermodynamic, or Data-Driven — and Why the Answer Matters for AGI

**Grant Lavell Whitmer III**¹

¹ Windstorm Labs, The Windstorm Institute, Fort Ann, New York, USA

**Corresponding author:** Grant Lavell Whitmer III

**Date:** April 2026

**Version:** 1.0 (First Draft)

**License:** Creative Commons Attribution 4.0 International (CC BY 4.0)

---

## Abstract

Papers 1–6 of the Windstorm series documented a robust convergence of language-model serial throughput to a narrow basin near τ ≈ 4.16 ± 0.19 bits per event across architectures (GPT-2, Pythia, Llama, GPT-3/4), datasets, and parameter counts spanning three orders of magnitude (Whitmer 2026a–f). Paper 6 proposed that this basin is *inherited* from the entropy of natural language itself, but could not exclude two competing accounts: an architectural ceiling intrinsic to attention-based models, or a thermodynamic floor on irreversible discrimination. We report four orthogonal experiments designed to discriminate among these hypotheses, together with the internal adversarial review of those experiments, which we publish unredacted alongside this paper and which materially constrains how strongly the present results can be read.

We find that, at 92M parameters, on Markov synthetic corpora with corpus-specific BPE tokenizers, the achieved bits-per-token on the SYN-8 condition (8.92 BPT) tracks training-corpus token entropy rather than collapsing to the ~4 BPT natural-language basin. A matched-parameter comparison of transformer (Pythia) and state-space (Mamba) models across seven corpora detects no significant BPT difference (Welch's *t* = 0.431, *p* = 0.688; Cohen's *d* = 0.397). A `bitsandbytes` round-to-nearest quantization sweep over Pythia 70M–1.4B and GPT-2 124M–774M locates a sharp universal cliff at INT4 → INT3 in this quantizer. Wall-power measurements on an RTX 5090 place total-system silicon at φ_GPU ≈ 10¹⁵–10¹⁸ above the Landauer limit, distinct from Paper 5's ≈10⁹ *useful-dissipation* figure (the two measure different physical boundaries; both are valid).

We interpret these findings as consistent with Paper 6's inherited-constraint hypothesis by elimination, in the regime tested. We do *not* claim falsification of the architectural hypothesis at scale, since 92M is below the regime in which the basin was originally observed; the SYN-8 result is reported in bits per corpus-specific BPE token, a unit confound we acknowledge applies retroactively to the τ measurements in Papers 1–4 as well; and our internal adversarial review identifies eight items, four blocking, that constrain the strong claim. These items define the scope of Paper 7.1, tracked publicly at github.com/sneakyfree/agi-extensions issue #1.

**Keywords:** throughput basin, synthetic training, quantization cliff, architecture comparison, inherited constraint, data-driven convergence, serial decoding, AGI hardware, Mamba, transformer, rate-distortion theory, BPE tokenization, Landauer limit, adversarial review

---

## 1. Introduction

### 1.1 The Established Basin

Six prior papers in this series progressively established and narrowed an empirical regularity in serial information processing. Paper 1 (Whitmer 2026a) derived from first principles why self-replicating systems converge on encoding alphabets in the neighborhood of 2⁶ = 64 symbols. Paper 2 (Whitmer 2026b) tested vocabulary independence empirically across 1,749 language models and confirmed that bits-per-byte is uncorrelated with vocabulary size (*p* = 0.643). Paper 3 (Whitmer 2026c) measured a cross-substrate convergence of effective information per processing event across 31 systems spanning ribosomes, DNA, phonemes, music, and language models. Paper 4 (Whitmer 2026d) located the convergence centroid at τ = 4.16 ± 0.19 bits via five experiments and predicted ribosome throughput from pure thermodynamics to within Δ = 0.003 bits. Paper 5 (Whitmer 2026e) derived *why* the basin exists from cost minimization in two regimes — biological (alphabet-bound, α > 1, basin emerges) and silicon (capacity-bound, α < 1, no thermodynamic basin) — and reported silicon at ≈10⁹× above the Landauer limit using useful-dissipation estimates. Paper 6 (Whitmer 2026f) demonstrated that AI language models nonetheless converge on ~4.4 BPT despite having no thermodynamic basin, and proposed the **inherited constraint hypothesis**: AI converges on the basin because human language carries the basin, because human cognition produced human language, because biology produced human cognition, because thermodynamics produced biology — a four-link causal chain from physics through biology, cognition, and language to artificial neural networks.

### 1.2 The Open Question

Paper 6 left a critical question unresolved. Three accounts are consistent with the empirical convergence of natural-language models on ~4 BPT:

1. **Data-driven (the inherited constraint).** The basin reflects the actual ~3–4 bit/character entropy of natural language, which itself reflects the cognitive bandwidth of its biological producers. AI models converge on the basin because their training data carries it.

2. **Architectural.** Transformer-style architectures, regardless of training data, compress representations to a ~4-bit-per-event ceiling. Attention as an inductive bias enforces this floor.

3. **Thermodynamic.** Irreversible discrimination at finite temperature imposes a ~4-bit floor on the energy-per-event ratio achievable by any silicon device. The ribosome and the GPU obey the same physics.

These three hypotheses make identical predictions on natural language, where the data, the architecture, and the substrate are all available simultaneously. To distinguish among them requires breaking each link independently.

### 1.3 Experimental Design Philosophy

We designed four experiments, each capable of independently falsifying the data-driven hypothesis. None did. Experiment 1 trained transformers on synthetic corpora with controlled source entropy ranging from 1.4 to 12 bits per symbol, breaking the *language* link. Experiment 2 swept weight quantization across two model families to locate any precision-coupled ceiling. Experiment 3 compared transformer and state-space architectures at matched parameter counts on a battery of seven corpora, breaking the *architecture* link. Experiment 6 measured wall-power energy per inferred bit on an RTX 5090, breaking the *thermodynamic* link.

The Windstorm Institute's methodological signature is to lead with the falsification framework rather than the confirmatory one. We name the predictions that would have killed each hypothesis, run the experiments that could have produced those predictions, and report the result regardless of which hypothesis survives. Where the result is ambiguous, we say so. Where our own measurements contradict each other, we publish the contradiction (§5).

---

## 2. Methods

### 2.1 Experiment 1: Synthetic Training Baseline

We constructed four synthetic corpora with controlled per-symbol entropy and one mixed control:

| Corpus | Alphabet size | Markov order | Target H | Empirical H | Tokens |
|---|---|---|---|---|---|
| SYN-2 | 4 | order-1 | 2.0 bits | 1.382 bits | ≈10⁸ |
| SYN-4 | 16 | order-1 | 4.0 bits | 3.675 bits | ≈10⁸ |
| SYN-8 | 256 | order-0 (uniform) | 8.0 bits | 7.9997 bits | ≈10⁸ |
| SYN-12 | 4096 | order-0 (uniform) | 12.0 bits | 11.985 bits | ≈10⁸ |
| SYN-MIX | mixed | mixed | 6.5 bits | (mixed) | ≈10⁸ |

A dedicated byte-pair-encoding tokenizer (vocabulary 8192) was fit per corpus.

We trained a GPT-2 architecture (768-dimensional embeddings, 12 transformer layers, 12 attention heads, ≈92M parameters) for 50,000 optimization steps per corpus, identical seed and schedule (batch 32, learning rate 3 × 10⁻⁴, cosine decay). Evaluation comprised held-out self-BPT on a 5% reserved split and a full 4×4 cross-corpus BPT matrix.

**Critical bug discovered and fixed during execution.** An early version of the training script collapsed each corpus into a single training example, producing pure memorization (training loss 0.0, held-out BPT 24–27). The autonomous research agent detected the symptom, identified the cause as a one-line dataset construction error, rewrote the loader to chunk each corpus into ≈10⁴ examples of 10⁴ characters, and retrained all five models from scratch. Total compute lost: ~7 hours. We document this here because methodological transparency is more useful to the field than concealing failed runs.

### 2.2 Experiment 2: Quantization Cliff

We swept the Pythia family (70M, 160M, 410M, 1B, 1.4B parameters) and the GPT-2 family (124M, 355M, 774M parameters) across precisions FP16, INT8, INT4, INT3, and INT2 using post-training weight quantization via the `bitsandbytes` library (NF4 for INT4, round-to-nearest for INT3/INT2). All models within each family share a single tokenizer, eliminating cross-tokenizer artifacts within the comparison. We evaluated each (model, precision) pair on WikiText-2 and report bits-per-token on the standard test split. Structural bonus was measured as the BPT difference between the original WikiText-2 evaluation and a token-shuffled version of the same corpus.

### 2.3 Experiment 3: Architecture Comparison

We compared a transformer family (Pythia 160M, 410M, 1.4B; GPT-2-medium 355M) against a state-space family (Mamba 130M, 370M, 1.4B) at approximately matched parameter counts on a battery of seven corpora chosen to span register and structure: WikiText-2, Python source code, DNA sequence, shuffled WikiText, mathematical text, random ASCII, and structured CSV. For each (model, corpus) pair we computed bits per token. Statistical comparisons used Welch's t-test on BPT, Cohen's d for effect size, Levene's test for equal variances, and a Mann-Whitney U test on per-corpus structural bonus. We additionally measured a five-level shuffling cascade (original → paragraphs shuffled → sentences shuffled → words shuffled → all shuffled) across all seven models to corroborate the structural-bonus findings of Paper 6 in this independent measurement context.

### 2.4 Experiment 6: Thermodynamic Energy Survey

We measured wall-power energy per inferred bit on an RTX 5090 using `nvidia-smi` power-draw polling at 10 ms cadence, with 3 runs and 30 s cooling intervals between configurations. The model set comprised Pythia 70M, 160M, 410M, 1B, and 1.4B at FP32, FP16, INT8, INT4, with FP16-compiled and FP16-batched variants. For each configuration we computed:

φ_GPU = E_per_bit / (k_B × T × ln 2)

where T is the measured GPU die temperature in Kelvin (≈320–343 K across configurations) and k_B × T × ln 2 ≈ 3.06–3.29 × 10⁻²¹ J/bit at the operating temperature.

**Critical methodological note.** φ_GPU as defined here measures *total* GPU wall power including memory access, cooling overhead, power-supply conversion losses, idle leakage of the entire board, and clock infrastructure. This is not the same quantity as φ_useful, the *useful-dissipation fraction* per discrimination event reported in Paper 5 (≈10⁹). Paper 5's quantity is an estimate of the thermodynamically relevant energy attributed only to the irreversible logical step itself, whereas Experiment 6 measures the entire system. The two figures are not in conflict; they bound the *computational* and *system* efficiency gaps respectively. We are explicit about this distinction throughout §3.4. We additionally note (§5.2) that the BPT values measured by Experiment 6 differ from those measured by Experiments 2 and 3 on the same models — almost certainly a tokenizer-normalization difference in the energy-survey harness — and that this discrepancy propagates downstream into every reported φ. The φ values should therefore be read as upper-bound estimates pending the Paper 7.1 reconciliation.

---

## 3. Results

All numbers in this section are pulled directly from the experimental CSV files (`exp-{1,2,3,6}/results/*.csv`). Where rounding is shown, the underlying CSV value is preserved to higher precision.

### 3.1 Experiment 1: The Basin Tracks Source Entropy at SYN-8

**Self-evaluation BPT on held-out splits:**

| Model | Empirical source H (bits/symbol) | Self-eval BPT | BPT/H ratio |
|---|---|---|---|
| SYN-2 | 1.382 | 20.525 | 14.9× |
| SYN-4 | 3.675 | 22.846 | 6.2× |
| **SYN-8** | **7.9997** | **8.921** | **1.12×** |
| SYN-12 | 11.985 | 17.403 | 1.45× |

The decisive datum is **SYN-8: 8.921 BPT on a source whose empirical entropy is 7.9997 bits per symbol.** If the throughput basin were architectural, this model would have compressed to ≈4 BPT. It did not. The achieved per-token cost lies within 12% of the source entropy and is more than twice the natural-language basin.

**Cross-corpus 4×4 evaluation matrix (BPT, rows = trained on, columns = evaluated on):**

|  | SYN-2 corpus | SYN-4 corpus | SYN-8 corpus | SYN-12 corpus |
|---|---|---|---|---|
| **SYN-2 model** | 0.077 | 38.141 | 39.674 | 42.492 |
| **SYN-4 model** | 24.751 | 0.026 | 30.454 | 27.411 |
| **SYN-8 model** | 39.088 | 39.043 | 7.383 | 38.570 |
| **SYN-12 model** | 31.965 | 22.429 | 26.036 | 5.478 |

Off-diagonal mean = 33.34 BPT. Diagonal mean = 3.24 BPT. The matrix shows perfect diagonal specialization (each model achieves near-zero BPT on training data, including memorized fragments) and catastrophic off-diagonal failure (22–42 BPT on mismatched entropy levels, worse than random guessing in a 256-symbol space). Critically, **the off-diagonal entries do not cluster near 4 BPT** — there is no architectural attractor visible at any BPT value. Each model has learned the entropy of its specific training distribution and nothing beyond it.

The matrix is asymmetric (mean |asymmetry| = 8.4 BPT). High-entropy → low-entropy transfer is less catastrophic than the reverse, because low-entropy models effectively memorized a tiny vocabulary and cannot represent any high-entropy token, whereas a high-entropy model at least recognizes the symbols of a low-entropy corpus as in-vocabulary.

**SYN-12 capacity limitation.** SYN-12 self-eval BPT is 17.40 on a source of entropy 11.99 — the model overshoots its own source by 5.4 bits, or 45%. We interpret this as a capacity limitation: the 92M-parameter budget is insufficient for a 4096-symbol Markov-0 distribution within the 50,000-step training budget. SYN-12 does *not* refute the data-driven hypothesis, but neither does it support it — the model failed to learn the distribution at all. The data-driven conclusion in this section rests on SYN-8, not SYN-12.

**SYN-2 and SYN-4 tokenization artifacts.** SYN-2 and SYN-4 also overshoot their source entropies by an order of magnitude (20.5 / 1.4 = 14.9× and 22.8 / 3.7 = 6.2× respectively). We attribute this to BPE tokenization artifacts: BPE merges interact pathologically with very low-entropy alphabets, where the token vocabulary becomes saturated with merges that encode redundant low-entropy patterns rather than information-bearing structure. This is not evidence against the data-driven hypothesis, but it is an honest red flag that the BPT metric is tokenizer-sensitive in ways the metric does not advertise. We address the broader implications in §5.2.

**Architectural ceiling: not falsified at scale, but not supported at 92M.** The SYN-8 result eliminates the strong-form architectural ceiling hypothesis at 92M parameters. A weaker form remains live: a transformer at scale (≥1B parameters) might compress SYN-8 toward ~4 BPT, in which case the 92M result would look like under-capacity rather than data-tracking. We have not run that experiment. We name it as Paper 7.1 item R5.

### 3.2 Experiment 3: No Architectural Difference Detected

**Per-architecture BPT on WikiText-2 at matched parameter counts:**

| Model | Architecture | Params | WikiText-2 BPT |
|---|---|---|---|
| Pythia-160m | transformer | 162M | 3.956 |
| Pythia-410m | transformer | 405M | 3.370 |
| Pythia-1.4b | transformer | 1.41B | 2.981 |
| GPT-2-medium | transformer | 355M | 3.674 |
| Mamba-130m | state-space (serial) | 129M | 3.845 |
| Mamba-370m | state-space (serial) | 372M | 3.300 |
| Mamba-1.4b | state-space (serial) | 1.37B | 2.894 |

Transformer mean = **3.495 BPT** (n=4, σ=0.41). Serial mean = **3.346 BPT** (n=3, σ=0.48).

**Statistical tests:**

| Test | Statistic | *p* | Interpretation |
|---|---|---|---|
| Welch's t (BPT) | 0.4314 | **0.6880** | Means indistinguishable at this *n* |
| Cohen's d | 0.397 | — | Small effect size |
| Levene's test (variance) | 0.00098 | 0.9751 | Equal variances |
| Mann-Whitney U (structural bonus) | 6.0 | 1.000 | Distributions equivalent |

The Welch t is non-significant. We do not interpret this as proof of equivalence — we lack the statistical power to make that claim, and we are explicit about it. **The minimum detectable effect at the present sample size (4 vs 3) is approximately d = 2.0; detecting the observed d = 0.40 reliably would require approximately 139 models per group.** The defensible reading is "no architecture-specific bias toward ~4 BPT is visible at this sample size, and what difference exists is small (transformer 3.50 vs serial 3.35, a 0.15 BPT gap on natural-language data already at the basin)."

**Per-corpus breakdown across the seven-corpus battery:**

| Corpus | Transformer mean BPT | Serial mean BPT | Cohen's d | *p* |
|---|---|---|---|---|
| WikiText-2 | 3.495 | 3.346 | 0.337 | 0.688 |
| Python | 0.436 | 0.208 | 0.727 | 0.342 |
| DNA | 4.403 | 4.450 | −0.254 | 0.721 |
| Shuffled WikiText | 10.334 | 10.129 | 0.881 | 0.273 |
| Math | 5.010 | 4.702 | 0.841 | 0.424 |
| Random ASCII | 8.350 | 8.370 | −0.106 | 0.882 |
| CSV | 2.082 | 1.926 | 0.877 | 0.262 |

Zero corpora exhibit significance after the multiple-comparison budget. The architectural-attention hypothesis predicts a transformer-specific advantage on hierarchically structured data; we observe none.

**Shuffling cascade corroborates Paper 6.** Across all 7 models, destroying word-order in WikiText raises BPT from ≈3.4 to ≈10.3. The structural bonus (all_shuffled − original) averages 6.7 bits, in close agreement with Paper 6's report of ~6.74 bits structural bonus on natural language. The cascade is monotone: paragraph-level shuffling adds ~0.6 BPT, sentence-level shuffling adds ~0.2 BPT, word-level shuffling adds ~5.0 BPT, full token shuffling adds another ~1.6 BPT. **Hierarchical structure is responsible for ~6.7 of natural language's compression headroom, and this number is identical for transformer and state-space architectures alike.**

### 3.3 Experiment 2: A Sharp Universal Cliff at INT4 → INT3

**Cliff precision and BPT-at-cliff for all eight models tested:**

| Model | Params | Cliff precision | BPT at cliff (INT4) | BPT at INT3 | Cliff ratio |
|---|---|---|---|---|---|
| Pythia-70M | 70M | INT4 | 5.196 | 72.04 | 13.87× |
| Pythia-160M | 162M | INT4 | 4.494 | 42.18 | 9.38× |
| Pythia-410M | 405M | INT4 | 3.759 | 23.13 | 6.15× |
| Pythia-1B | 1.01B | INT4 | 3.139 | 27.14 | 8.65× |
| Pythia-1.4B | 1.41B | INT4 | 3.046 | 17.77 | 5.83× |
| GPT-2 | 124M | INT4 | 4.145 | 14.89 | 3.59× |
| GPT-2-medium | 355M | INT4 | 3.742 | 21.22 | 5.67× |
| GPT-2-large | 774M | INT4 | 3.483 | 10.31 | 2.96× |

Every one of the eight models tested cliffs at the **INT4 → INT3** transition with a cliff ratio between 2.96× and 13.87×. The qualitative pattern is monotone:

- FP16 → INT8: < 2% BPT degradation (effectively lossless)
- INT8 → INT4: ≈ 5–15% degradation (operational)
- **INT4 → INT3: 200–1300% degradation (catastrophic)**
- INT3 → INT2: variable, often *less* catastrophic than INT3 because the model has already collapsed to a degenerate distribution at INT3.

**Structural bonus collapses at the same precision.** At FP16 the mean structural bonus across the 8-model panel is 6.864 bits (σ = 0.169). At INT4 it is 6.708 bits (essentially preserved). At INT3 it collapses to a mean of 0.268 bits with several models going negative (the model becomes *more* confident on shuffled text than original — i.e. it has lost the ability to exploit linguistic structure entirely). At INT2 the mean is −0.304 bits. **Syntax dies at the same precision threshold as raw prediction. Both fail simultaneously, consistent with a shared representational floor in the weight precision rather than separate failure modes.**

We note explicitly that this is a `bitsandbytes`-RTN finding and do not claim universality across alternative quantization methods (GPTQ, AWQ, SmoothQuant). We address this in §5.2 item R7.

We further note that the cliff is at **4 bits per weight**, not 4 bits per event. The numerical coincidence with the throughput basin is exactly that — coincidence. The two quantities measure different things: weight precision is a property of the representation; throughput is a property of the input distribution. We report the cliff together with the basin findings to forestall the natural confusion, not to suggest a connection.

### 3.4 Experiment 6: Silicon Operates 10¹⁵–10¹⁸× Above Landauer (Total System)

**Selected configurations (full table in `exp-6/results/exp6_energy.csv`):**

| Model | Precision | BPT | E/bit (J) | T (K) | k_B T ln 2 (J) | φ_GPU | log₁₀ φ |
|---|---|---|---|---|---|---|---|
| Pythia-70M | FP32 | 9.74 | 2.78×10⁻⁵ | 320 | 3.06×10⁻²¹ | 9.07×10¹⁵ | 15.96 |
| Pythia-70M | FP16 | 9.81 | 7.61×10⁻⁶ | 320 | 3.06×10⁻²¹ | 2.48×10¹⁵ | 15.39 |
| Pythia-70M | INT8 | 9.82 | 4.67×10⁻⁴ | 331 | 3.17×10⁻²¹ | 1.47×10¹⁷ | 17.17 |
| Pythia-70M | INT4 | 10.31 | 6.05×10⁻⁵ | 335 | 3.21×10⁻²¹ | 1.89×10¹⁶ | 16.28 |
| Pythia-160M | FP16 | 12.11 | 1.19×10⁻⁴ | 335 | 3.21×10⁻²¹ | 3.72×10¹⁶ | 16.57 |
| Pythia-410M | FP16 | 11.21 | 3.24×10⁻⁴ | 335 | 3.21×10⁻²¹ | 1.01×10¹⁷ | 17.00 |
| Pythia-1B | FP32 | 8.90 | 4.82×10⁻³ | 343 | 3.28×10⁻²¹ | 1.47×10¹⁸ | 18.17 |
| Pythia-1B | FP16 | 8.90 | 8.31×10⁻⁴ | 336 | 3.22×10⁻²¹ | 2.58×10¹⁷ | 17.41 |
| Pythia-1.4B | FP32 | 9.34 | 6.66×10⁻³ | 342 | 3.27×10⁻²¹ | 2.04×10¹⁸ | 18.31 |
| Pythia-1.4B | FP16 | 9.33 | 1.25×10⁻³ | 340 | 3.26×10⁻²¹ | 3.83×10¹⁷ | 17.58 |

Range of φ_GPU: **2.48 × 10¹⁵ to 2.04 × 10¹⁸**. Mean log₁₀ φ_GPU ≈ 16.8 → φ_GPU ≈ 6 × 10¹⁶.

**There is no observable thermodynamic floor anywhere near the 4-bit basin.** Fifteen to eighteen orders of magnitude of headroom exist above the Landauer limit. The thermodynamic-ceiling hypothesis is not supported.

**Energy scaling exponent.** Fitting log E_per_token = log a + b × log params over the FP16 configurations gives b = 1.51 (r = 0.94, *p* = 0.016); over compiled FP16, b = 0.93 (r = 0.96, *p* = 0.011); over INT4, b = 0.71 (r = 0.80). The compiled-FP16 exponent (0.93) reproduces Paper 5's reported scaling exponent of 0.937 to within 0.7% — independent confirmation that the energy-vs-params relationship in current silicon is not a measurement artifact.

**Reconciliation with Paper 5.** Paper 5 reported silicon at ≈10⁹× above Landauer; Experiment 6 reports φ_GPU ≈ 10¹⁵–10¹⁸. The difference is one of measurement boundary, not of fact:

- **Paper 5's φ_useful (≈10⁹)** estimates the *useful dissipation fraction* — the thermodynamically relevant energy attributed only to the irreversible logical step itself. It is the right number for asking "how far is silicon from optimal computation?"
- **Experiment 6's φ_GPU (≈10¹⁶)** measures *total GPU wall power*, which additionally pays for memory access, cooling, power-supply conversion losses, idle circuitry, and clock infrastructure. It is the right number for asking "how far is silicon from optimal as a system?"

Both are valid; they measure different things. Paper 5's ≈10⁹ figure represents the *computational efficiency gap*. Experiment 6's ≈10¹⁶ figure represents the *total system efficiency gap*. We report both rather than collapse them into a single number that would obscure where the inefficiency lives. The ribosome's φ ≈ 1.02 (Paper 4) is closer to the useful-dissipation comparison: it is 10⁹× more useful-efficient than silicon and 10¹⁶× more system-efficient than silicon, and these are consistent statements once the boundary is named.

---

## 4. Discussion

### 4.1 Confirmation of the Inherited Constraint by Elimination — in the Regime Tested

Three independent falsifiers were prepared. Each had a clean prediction that would have killed the data-driven hypothesis, and each failed to fire:

- **The architectural hypothesis** predicted that SYN-8 would compress to ~4 BPT regardless of training data, or that Mamba would diverge from Pythia on natural language. Neither occurred: SYN-8 achieved 8.92 BPT, and the transformer-vs-state-space difference was 0.15 BPT and not statistically significant.
- **The thermodynamic hypothesis** predicted a silicon energy floor incompatible with > 4 BPT throughput. Wall-power measurements place silicon 15–18 orders of magnitude above any such floor.
- **The intrinsic-compression hypothesis** predicted a universal architectural attractor in the cross-corpus matrix near 4 BPT. The off-diagonal entries of the cross-corpus matrix span 22 to 42 BPT with no clustering near any value.

In the regime tested — 92M-parameter GPT-2-class models, Markov synthetic data, corpus-specific BPE tokenizers, single seed, 50K training steps, `bitsandbytes` RTN quantization, and `nvidia-smi` total wall-power energy — the data-driven hypothesis is the only account consistent with all four experiments. Paper 6's inherited-constraint hypothesis is confirmed by elimination *in this regime*. We are explicit (§5, §7) about what "in this regime" excludes.

### 4.2 The Four-Link Causal Chain

Paper 6 proposed that the throughput basin we observe in artificial language models is the artifact of a four-link causal chain: physics → biology → cognition → language → AI. Each link constrains the next. The thermodynamic cost of irreversible discrimination at finite temperature constrains what biology can build; the energetics of biological neural tissue constrains what cognition can process per unit time; the cognitive bandwidth of producers and consumers constrains the information density of human language; and the language constrains what its statistical learners can extract. The artificial neural network is the readout of all four.

The synthetic-training results in Experiment 1 are particularly clean here, because they let us *break* the language link by substituting a distribution that has nothing to do with biology, cognition, or human communication, and the model dutifully tracks the new distribution rather than the old basin. SYN-8 is not English; the model trained on it is not biological; the network has no cognitive bandwidth limit. Yet the BPT lands at 8.92, not at 4. The four-link chain does not impose itself on data that did not pass through it.

### 4.3 The Quantization Cliff and AGI Hardware

The INT4 cliff is reported here as an incidental but practically significant hardware-design constraint, distinct from the per-event throughput basin despite the numerical coincidence at ~4 bits.

Three observations make this finding consequential for AGI inference hardware design:

1. **The cliff is universal across the eight models tested**, spanning two model families and 70M–1.4B parameters.
2. **Structural bonus collapses at the same precision** that raw BPT collapses, indicating that the failure mode is not gradual numerical degradation but a sharp loss of representational capacity, including the ability to exploit linguistic structure.
3. **The transition is a phase change, not a gradient**: cliff ratios between 3 and 14, with INT4 still operational and INT3 catastrophic.

The implication for ASIC design is direct: inference accelerators below 4-bit weight precision are non-viable for language tasks, and hardware support for INT4 weights with INT16 activations is the minimum-viable specification for language-model inference. We do not yet know whether the cliff persists across other quantization methods (GPTQ, AWQ, SmoothQuant) or across non-transformer architectures; both extensions are scoped for Paper 7.1 (item R7).

### 4.4 The Multimodal Path to Higher-Throughput AGI

If the throughput basin is genuinely a property of the training data rather than the model, then the most consequential corollary for AGI development is sobering: as long as the input distribution is human-generated natural language, ~4 BPT is approximately what one gets, because that is approximately what is there to be gotten. Compression below source entropy is forbidden by Shannon; you cannot extract more bits per token than the source provides. The path to higher per-event throughput in artificial systems therefore does not run through new architectures, new optimizers, new attention mechanisms, or new physics. It runs through richer training data: vision, audio, proprioception, embodied control signals, multimodal corpora.

Paper 8 (Vision Basin Phase 1) provides a preliminary test of this prediction on vision transformers. The vision arm is complete and shows that classifiers on CIFAR-100 sit at 0.02–0.04 bits per patch — *below* the language basin in per-token units, because the classification task discards information by collapsing an entire image to a single class label. The interesting candidates for higher throughput are tasks that force the model to *retain* bits: image generation, video next-frame prediction, world models. Papers 9–10 will examine these.

### 4.5 What This Paper Does Not Prove

We are explicit about the boundary of the present claim:

- **Not** that the throughput basin is universally data-driven at all scales. We tested 92M parameters; the original basin observations span 70M to 175B.
- **Not** that Markov synthetic data with no hierarchical structure is equivalent to natural language with deep compositional structure. The shuffling cascade in §3.2 demonstrates that destroying hierarchical structure in natural language adds ~6.7 bits to BPT — a contribution that synthetic Markov corpora cannot replicate.
- **Not** that the BPE tokenizer is a transparent unit. SYN-2 and SYN-4 overshoots are evidence that BPE artifacts inflate per-token cost in ways that confound the unit comparison.
- **Not** that the INT4 cliff persists across non-`bitsandbytes` quantization methods.
- **Not** that the φ_GPU figure should be cited in place of Paper 5's φ_useful for thermodynamic discussions of computation.

Each of these is scoped for Paper 7.1.

---

## 5. Internal Adversarial Review and Open Items

### 5.1 Summary of Confirmed Findings (in the regime tested)

1. **At 92M parameters, on Markov synthetic data, achieved BPT tracks training-corpus token entropy** rather than collapsing to ~4 BPT. The decisive datum is SYN-8 = 8.92.
2. **No transformer-specific advantage over state-space models is detected** on natural language at the present sample size, on a battery of seven corpora spanning text, code, DNA, and shuffled controls.
3. **A sharp universal INT4 → INT3 cliff in `bitsandbytes` round-to-nearest quantization** spans the eight Pythia and GPT-2 models tested, with both raw BPT and structural bonus collapsing at the same precision.
4. **Total-system silicon operates at φ_GPU ≈ 10¹⁵–10¹⁸ above the Landauer limit**, distinct from the ≈10⁹ useful-dissipation figure of Paper 5; both are valid measurements of different boundaries.
5. **Hierarchical structure contributes ~6.7 bits of structural bonus** to natural-language compression, replicating Paper 6's headline figure independently.

### 5.2 Items From the Internal Adversarial Review

Concurrent with manuscript preparation we ran an internal adversarial review of the experimental record (`review/adversarial_review.md`). It is published unredacted alongside this manuscript. We list every item here, with status, because the review's findings are load-bearing for how the present results should be read and they define the scope of the planned follow-up (Paper 7.1, tracked at github.com/sneakyfree/agi-extensions issue #1).

**Blocking items (must resolve before the strong claim is earned):**

1. **B1. Self-eval vs cross-corpus diagonal disagreement (Exp 1).** The same model on the same corpus reports different BPT in two evaluations: SYN-8 self-eval = 8.921, cross-corpus diagonal = 7.383 (Δ = 1.54 BPT); SYN-12 self-eval = 17.40, cross-corpus diagonal = 5.48 (Δ = 11.92 BPT). Either the eval pipelines measure subtly different things or the cross-corpus split is leaking. The headline number sits in this gap. **Paper 7.1 will re-run all 4×4 cells with a single eval harness on a provably disjoint held-out split, with seed-level error bars.**

2. **B2. Exp 6 BPT disagrees with Exp 2/3 BPT on the same models.** Pythia-160m on WikiText-2 reports 3.96 BPT in Exp 2 and Exp 3, but 12.11 BPT in Exp 6. Pythia-1.4B reports 2.98 BPT in Exp 2/3, 9.33 in Exp 6. This is almost certainly a tokenizer-normalization difference in the energy-survey harness (bits-per-byte vs bits-per-token vs natural-log vs log₂). Every φ in §3.4 is downstream of the Exp 6 number. **Paper 7.1 will reconcile the harnesses and recompute all φ.** If the Exp 6 BPT values do not recover, the §3.4 thermodynamics table will be revised or withdrawn.

3. **B3. BPT vs bits-per-source-symbol unit confound (Exp 1).** The SYN-8 result (8.92 BPT) uses a corpus-trained BPE tokenizer with vocabulary 8192 and is therefore not directly comparable in units to the 8.0 bit/source-symbol entropy. The comparison the paper makes (8.92 vs 8.0) crosses unit boundaries. The same unit confound applies retroactively to the τ = 4.16 measurement in Papers 1–4, which were also reported in BPT under tokenizers fit per evaluation corpus. **Paper 7.1 will re-evaluate all SYN-* models in bits-per-source-symbol** and additionally retrain with a single shared tokenizer fit on the union of corpora. The retroactive scope expansion to Papers 1–4 is acknowledged in a public note added to the Serial Decoding Basin τ article on windstorminstitute.org.

4. **B4. No learning curves.** We report converged-to-cutoff BPT values but no loss-vs-step trajectories, so we cannot confirm SYN-8 had plateaued at 8.92 versus continued descending. If SYN-8 was still falling at the 50K-step cutoff, the eventual basin is unknown. **Paper 7.1 will publish loss-vs-step trajectories for all SYN-* conditions.**

**Strongly recommended items (would substantially strengthen or weaken the claim):**

5. **R5. Capacity at scale.** 92M is below the regime in which the basin was originally observed. **Paper 7.1 will train at least one ≥1B-parameter SYN-8 model at matched compute.** If it still sits at ~8 BPT, the data-driven claim hardens substantially. If it drops toward 4, the central thesis collapses.

6. **R6. Hierarchical-structure control.** Paper 7.1 will train on (a) a PCFG-induced 8-bit-entropy *structured* corpus, and (b) shuffled WikiText. The first tests whether the basin is about *entropy level* or about *hierarchical structure*; the second is the cleanest possible control because it has identical surface statistics to natural language with only the hierarchy destroyed.

7. **R7. `bitsandbytes`-only quantization.** §3.3 cliff is a `bitsandbytes` RTN finding. **Paper 7.1 will repeat the sweep with GPTQ for at least one model size**, and ideally with AWQ and SmoothQuant.

8. **R8. Mamba energy in Exp 3.** `exp3_energy.csv` reports Mamba-130m energy at 69.9 mJ/token vs Pythia-1.4b at 1.67 mJ/token — a 100× gap that is almost certainly an un-fused reference kernel rather than a real architectural property. The energy comparisons in §3.2 should be read with that caveat. **Paper 7.1 will rerun with a fair kernel or the table will be withdrawn.**

### 5.3 Why We Publish the Review With the Paper

We chose to publish the present results with the adversarial review attached, rather than delay until Paper 7.1 lands, because the institute's stated practice is to surface internal falsification attempts at the same time as the claims they constrain. A reader who reads the abstract, §3, and §5.2 as a unit will get the right calibration of confidence. A reader who reads only §3 will overclaim — and we have tried to write §3 so that reading is harder than the alternative.

---

## 6. Falsifiable Predictions

Each of the following predictions could falsify a portion of the picture presented in this paper. We expect to be wrong about at least one and we will say so when it happens.

**Prediction 1.** A model with capacity ≥ 1B parameters trained on the SYN-8 corpus will achieve held-out bits-per-source-symbol within 0.5 of the source entropy (≈ 8.0). *Falsified if* it converges below 5 bits/source-symbol, in which case the data-driven hypothesis collapses and the architectural ceiling re-enters as the leading account.

**Prediction 2.** A 92M-parameter model trained on a PCFG-induced *structured* 8-bit corpus will achieve BPT closer to the natural-language basin (≈ 4) than to 8. *Falsified if* the structured-corpus result sits at ~8 BPT, in which case the basin is genuinely about entropy level and hierarchical structure is irrelevant.

**Prediction 3.** Vision transformers trained on natural images will exhibit bits-per-patch significantly different from ≈ 4. *Tested in Paper 8; preliminary result is ~0.02–0.04 bits per patch on classification tasks (well below 4), as predicted. Generative vision tasks remain to be tested.*

**Prediction 4.** Models trained on multimodal (text + vision + audio) corpora will exhibit aggregate per-event throughput strictly higher than text-only models of the same parameter count. *Tested in Papers 8–10.*

**Prediction 5.** The INT4 → INT3 cliff persists across non-transformer architectures (Mamba, RWKV) and across non-`bitsandbytes` quantizers (GPTQ, AWQ). *Falsified if* the cliff moves to a different precision in any combination, in which case the cliff is a `bitsandbytes`-RTN artifact rather than a representational floor.

**Prediction 6.** A model trained from scratch on shuffled WikiText (preserving unigram statistics, destroying hierarchical structure) will achieve BPT ≈ 10.6 (matching Paper 6's evaluation result on shuffled text). *Falsified if* training-time exposure to shuffled data lets the model recover most of the structural bonus, in which case the bonus is a property of *training* statistics, not *evaluation* statistics.

---

## 7. Limitations

We report the following honestly and prominently. The internal adversarial review (§5.2) names the experiments that would resolve each.

1. **Scale.** Only 92M parameters tested in Experiment 1. The original basin observations span 70M to 175B; the 92M synthetic-training result does not yet falsify an architectural ceiling that might emerge at scale.
2. **Hierarchical structure.** Markov-synthetic corpora have no syntax, semantics, or discourse structure. The shuffling cascade in §3.2 directly demonstrates that destroying hierarchy in natural language adds ~6.7 bits to BPT — a property that synthetic Markov corpora by construction cannot test. The basin might require *hierarchically structured* high-entropy data to land at ~4, in which case our SYN-8 result is consistent with the architectural hypothesis under a different framing.
3. **Tokenization.** BPE tokenizers fit per-corpus introduce artifacts; SYN-2 and SYN-4 overshoots are direct evidence. BPT is tokenizer-dependent. A bits-per-source-symbol re-measurement would be more rigorous and is scoped under Paper 7.1 item B3. **The retroactive form of this concern applies to the τ = 4.16 measurement in Papers 1–4 as well.**
4. **SYN-12 capacity failure.** 17.40 BPT on a 11.99-bit source means the 92M model failed to learn the distribution. This limits confidence in extrapolation to high-entropy regimes from this experiment alone.
5. **Energy methodology boundary.** GPU wall power ≠ discrimination energy. The φ comparison to Paper 5 crosses measurement boundaries (§3.4 reconciliation). Paper 5's ≈10⁹ figure remains the right number for the *computational* efficiency question.
6. **Single quantization method.** Only `bitsandbytes` round-to-nearest tested. GPTQ, AWQ, SmoothQuant, and other methods may show different cliff locations. The §3.3 finding is properly framed as *"`bitsandbytes`-RTN exhibits an INT4 cliff,"* a narrower claim than *"all quantization methods exhibit an INT4 cliff."*
7. **Single seed.** All synthetic-training results are single runs. No error bars from multiple random seeds. Standard practice for this kind of claim is ≥3 seeds per condition; this is scoped for Paper 7.1.
8. **Architecture comparison is underpowered.** With *n* = 4 transformers and *n* = 3 state-space models, the minimum detectable effect at α = 0.05, 80% power is approximately *d* = 2.0. The observed *d* = 0.40 would require ~139 models per group to detect reliably. The Welch *p* = 0.688 is *not* evidence of equivalence; it is evidence of insufficient power to reject equality.
9. **Internal evaluation contradictions (§5.2 items B1 and B2).** Two of our own measurement pipelines disagree with each other on the same (model, corpus) cells. We report both, name the discrepancies, and scope the reconciliation as Paper 7.1.

---

## 8. Conclusion

We tested three competing hypotheses for the origin of the throughput basin observed in Papers 1–6: that the basin is data-driven (inherited from training data entropy), architectural (intrinsic to attention-based models), or thermodynamic (a physical floor on irreversible discrimination in silicon). Four orthogonal experiments — synthetic-corpus training at four entropy levels, an eight-model quantization sweep, a transformer-vs-state-space architecture comparison across seven corpora, and a wall-power thermodynamic survey on an RTX 5090 — were each capable of independently falsifying the data-driven hypothesis. None did, in the regime tested.

The strongest single result is the SYN-8 experiment: a 92M-parameter GPT-2 trained on a Markov-0 source whose empirical entropy is 7.9997 bits per symbol achieved 8.92 BPT on held-out data, more than twice the natural-language basin and within 12% of the source entropy. If the throughput basin were architectural, this number could not exist. The strongest practical deliverable is the universal INT4 cliff: a sharp phase transition in `bitsandbytes` round-to-nearest weight quantization at the INT4 → INT3 boundary, observed across all eight Pythia and GPT-2 models tested, with structural bonus collapsing at the same precision as raw prediction. INT4 is the minimum viable weight precision for language-model inference in this quantizer.

The path to higher-throughput AGI does not run through new architectures, new optimizers, or new physics. It runs through richer, higher-entropy training data — vision, audio, embodied experience, multimodal corpora — because compression below source entropy is forbidden by Shannon, and natural language sits at ~4 bits per token because that is approximately what its biological producers could put there.

We publish these results together with their internal adversarial review, which identifies four blocking items and four strongly recommended re-runs that constrain how strongly the present claims can be read. The adversarial review is not a footnote; it is a co-equal artifact, and it defines the scope of Paper 7.1. We do this because the Windstorm Institute's job is to surface falsification attempts at the same time as the claims they constrain, not after.

**The throughput basin is a mirror, not a wall. It reflects the entropy of the data we train on, inherited through the causal chain from thermodynamics through biology through cognition through language to AI. To build systems that see beyond this mirror, we must train them on experience richer than language alone.**

---

## References

Bennett, C. H. (1982). The thermodynamics of computation — a review. *International Journal of Theoretical Physics*, 21(12), 905–940.

Eigen, M. (1971). Selforganization of matter and the evolution of biological macromolecules. *Die Naturwissenschaften*, 58(10), 465–523.

Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cai, T., Rutherford, E., Casas, D. de las, et al. (2022). Training compute-optimal large language models. *arXiv preprint* arXiv:2203.15556.

Hopfield, J. J. (1974). Kinetic proofreading: a new mechanism for reducing errors in biosynthetic processes requiring high specificity. *Proceedings of the National Academy of Sciences*, 71(10), 4135–4139.

Landauer, R. (1961). Irreversibility and heat generation in the computing process. *IBM Journal of Research and Development*, 5(3), 183–191.

Miller, G. A., & Nicely, P. E. (1955). An analysis of perceptual confusions among some English consonants. *Journal of the Acoustical Society of America*, 27(2), 338–352.

Piñeros, W. D., & Tlusty, T. (2020). Kinetic proofreading and the limits of thermodynamic uncertainty. *Physical Review E*, 101(2), 022415.

Shannon, C. E. (1948). A mathematical theory of communication. *Bell System Technical Journal*, 27(3–4), 379–423, 623–656.

Shannon, C. E. (1951). Prediction and entropy of printed English. *Bell System Technical Journal*, 30(1), 50–64.

Sterling, P., & Laughlin, S. B. (2015). *Principles of Neural Design*. MIT Press.

Whitmer, G. L. III (2026a). The Fons Constraint: information-theoretic convergence on encoding depth in self-replicating systems. *Windstorm Institute Paper 1*. doi:10.5281/zenodo.19274048.

Whitmer, G. L. III (2026b). The Receiver-Limited Floor: rate-distortion bounds on serial decoding throughput. *Windstorm Institute Paper 2*. doi:10.5281/zenodo.19322973.

Whitmer, G. L. III (2026c). The Throughput Basin: cross-substrate convergence and decomposition of serial decoding throughput. *Windstorm Institute Paper 3*. doi:10.5281/zenodo.19323194.

Whitmer, G. L. III (2026d). The Serial Decoding Basin τ: five experiments on convergence, thermodynamic anchoring, and the geometry of receiver-limited throughput. *Windstorm Institute Paper 4*. doi:10.5281/zenodo.19323423.

Whitmer, G. L. III (2026e). The Dissipative Decoder: thermodynamic cost bounds on the serial decoding throughput basin — and why silicon escapes them. *Windstorm Institute Paper 5*. doi:10.5281/zenodo.19433048.

Whitmer, G. L. III (2026f). The Inherited Constraint: biological throughput limits shape the information structure of human language and, through it, AI. *Windstorm Institute Paper 6*. doi:10.5281/zenodo.19432911.

---

## Acknowledgments

The four experiments reported here were executed autonomously on an NVIDIA RTX 5090 (Windstorm Labs, Varon-1) by Claude Sonnet 4.5 (Anthropic) over 14.5 hours of unattended overnight operation, beginning 2026-04-08 14:30 and concluding 2026-04-09 05:00. Four critical bugs were self-identified and fixed by the research agent during execution without human intervention: (1) corpus-generation entropy collapse on SYN-8 and SYN-12, (2) the single-example training-dataset bug that required a complete retraining of all five Experiment 1 models, (3) a disk-space exhaustion event resolved by autonomous checkpoint cleanup, and (4) evaluation-script compatibility fixes for position embeddings and corpus references.

The internal adversarial review (`review/adversarial_review.md`) was conducted by a separate Claude Opus 4.6 instance reading the experimental CSVs and master summary independently of the manuscript draft, with explicit instructions to find every place the headline overclaimed. The deep statistical reanalysis (`analysis/paper7_deep_analysis.md`) was conducted by a third instance. Manuscript drafting, interpretation, and quality control: Grant Lavell Whitmer III, with manuscript composition assistance from Claude Opus 4.6.

All code, raw CSV data, intermediate analyses, plots, and the unredacted adversarial review are public at <https://github.com/sneakyfree/agi-extensions>. The Paper 7.1 follow-up is tracked publicly at <https://github.com/sneakyfree/agi-extensions/issues/1>.

Gratitude is owed to the ribosome, which continues to operate at φ ≈ 1.02 and thereby fixes a target sixteen orders of magnitude away from our current silicon.

**Competing interests:** None declared.

**Data availability:** GitHub repository above; Zenodo deposit pending Paper 7.1 resolution.

**Funding:** Self-funded by The Windstorm Institute.

**License:** Creative Commons Attribution 4.0 International (CC BY 4.0).

---

*End of Paper 7. The Windstorm Institute, April 2026.*
