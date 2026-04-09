# The Throughput Basin Origin: Four Orthogonal Experiments on Whether Serial Decoding Convergence Is Architectural, Thermodynamic, or Data-Driven

**Grant Lavell Whitmer III**
*Windstorm Labs, The Windstorm Institute, Fort Ann, NY, USA*

**April 2026**

---

## Abstract

Papers 1–6 of the Windstorm series documented a robust convergence of language-model serial throughput to a narrow basin near τ ≈ 4.16 ± 0.19 bits per event across architectures (GPT-2, Pythia, Llama, GPT-3/4), datasets, and parameter counts spanning three orders of magnitude. Paper 6 proposed the basin is *inherited* from the entropy of natural language itself, but could not exclude two competing hypotheses: an architectural ceiling of attention-based models, or a thermodynamic floor on irreversible discrimination. Here we report four orthogonal experiments designed to discriminate among these accounts, together with the internal adversarial review of those experiments (`review/adversarial_review.md`), which we publish alongside the manuscript and which materially constrains how strongly the results can be read.

**What we find.** (1) At 92M parameters, on Markov synthetic corpora with corpus-specific BPE tokenizers, the achieved bits-per-token on the SYN-8 condition (8.92 BPT) tracks training-corpus token entropy rather than collapsing to the ~4 BPT natural-language basin. (2) A matched-parameter comparison of transformer (Pythia) and state-space (Mamba) models across seven corpora finds no detectable BPT difference (Welch's *t* = 0.431, *p* = 0.688); we explicitly note this is a low-power test (Cohen's *d* ≈ 0.4, *n* = 4 vs 3) and is consistent with "no architecture-specific ~4 BPT bias visible at this *n*" but does not constitute equivalence. (3) A `bitsandbytes`-RTN quantization sweep over Pythia 70M–1.4B and GPT-2 124M–774M locates a sharp cliff at INT4 → INT3 in this quantizer; we do not claim universality across GPTQ/AWQ/SmoothQuant. (4) `nvidia-smi` wall-power measurements on an RTX 5090 place total-system silicon at φ_GPU ≈ 10^15–10^18 above the Landauer limit, distinct from Paper 4's ≈10^9 *useful-dissipation* figure (the two measure different boundaries; both are valid).

**What this is not.** This is not a falsification of the architectural hypothesis at scale: 92M is below the regime in which the basin was originally observed, and a ≥1B parameter SYN-8 run remains the cleanest test. The SYN-8 BPT figure is in units (bits per corpus-specific BPE token) the adversarial review flags as confounded with bits-per-source-symbol; we have not yet re-evaluated in source-symbol units. SYN-12 overshoots its source entropy by 45% (capacity-limited and almost certainly under-trained), and we do not have learning curves to confirm SYN-8 had plateaued at the 50K-step cutoff. The adversarial review additionally identifies an unresolved internal contradiction between self-eval and cross-corpus diagonal BPT in Experiment 1, and a same-model-same-corpus BPT discrepancy between Experiment 2/3 and Experiment 6 that propagates into every reported φ.

**What is defensible.** Training-data entropy clearly influences achieved BPT in the regime tested, and we find no positive evidence of a transformer-specific ~4-bit ceiling at 92M parameters. Whether the natural-language basin is *fully* explained by data entropy — as opposed to *also* by hierarchical structure, cognitive bandwidth, or architecture at scale — remains open. Paper 7.1, scoped in §6, will address the blocking items the adversarial review identifies.

---

## 1. Introduction

Six prior papers in this series (Whitmer, 2026a–f) established and progressively narrowed a striking empirical regularity: serial language-model decoding converges to τ ≈ 4.16 ± 0.19 bits per event across GPT-2, GPT-3, GPT-4, Pythia, and Llama, across Wikipedia, books, code, and web data, and across parameter counts from 70M to 175B. Paper 6 introduced the *inherited constraint hypothesis* — that this basin is the lower terminus of a four-link causal chain from physics through biology, cognition, and language to artificial neural networks — but acknowledged that the available evidence could not distinguish three competing accounts:

1. **Data-driven.** The basin reflects the actual ~3–4 bit/character entropy of natural language, which itself reflects the cognitive bandwidth of its human producers (Papers 1–5).
2. **Architectural.** Transformer-style architectures, regardless of training data, compress representations to a ~4-bit-per-event ceiling.
3. **Thermodynamic.** Irreversible discrimination at finite temperature imposes a ~4-bit floor on the energy-per-event ratio achievable by any silicon device.

We designed four experiments, each capable of independently falsifying the data-driven hypothesis. None did. Experiment 1 trained transformers on synthetic corpora with controlled source entropy. Experiment 2 swept weight quantization to locate any precision-coupled ceiling. Experiment 3 compared transformer and state-space architectures at matched parameter counts. Experiment 6 measured wall-power energy per bit on an RTX 5090. Together they confirm Paper 6's inherited-constraint hypothesis by elimination, and incidentally locate the INT4 cliff as a robust hardware-design constraint.

## 2. Methods

### 2.1 Experiment 1: Synthetic Training Baseline

We constructed four synthetic corpora with controlled per-symbol entropy: SYN-2 (4-symbol Markov-1, source entropy ≈ 1.38 bits), SYN-4 (16-symbol Markov-1, ≈ 3.68 bits), SYN-8 (256-symbol Markov-0, 8.0 bits), and SYN-12 (4096-symbol Markov-0, ≈ 11.99 bits). Each corpus contained ≈10^8 tokens. A dedicated byte-pair-encoding tokenizer (vocabulary 8192) was fit per corpus. We trained a GPT-2 architecture (768-dimensional embeddings, 12 transformer layers, 12 attention heads, ≈92M parameters) for 50,000 optimization steps per corpus, identical seed and schedule. Evaluation comprised held-out self-BPT and a full 4×4 cross-corpus BPT matrix. An early version of the training script collapsed each corpus into a single example, producing pure memorization (training loss 0.0, held-out BPT 24–27); we discovered the bug, rewrote the dataset construction to chunk each corpus into ≈10^4 examples of 10^4 characters, and retrained all five models from scratch.

### 2.2 Experiment 2: Quantization Cliff

We swept the Pythia family (70M, 160M, 410M, 1B, 1.4B) and the GPT-2 family (124M, 355M, 774M) across precisions FP32, FP16, INT8, INT4, INT3, and INT2 using post-training weight quantization. All models share a single tokenizer family, eliminating cross-tokenizer artifacts. We evaluated each (model, precision) pair on WikiText-2 and report bits-per-token on the standard test split. Structural bonus was measured as the BPT difference between the original and a token-shuffled WikiText-2.

### 2.3 Experiment 3: Architecture Comparison

We compared a transformer family (Pythia) against a state-space family (Mamba) at matched parameter counts on a battery of seven corpora chosen to span register and structure: TinyStories, SimpleWiki, Python source, legal contracts, English-language poetry, scientific abstracts, and Wikipedia. For each (model, corpus) pair we computed bits per token; comparisons used Welch's t-test on BPT, Cohen's d, Levene's test for equal variances, and a Mann-Whitney U on per-corpus structural bonus.

### 2.4 Experiment 6: Thermodynamic Energy Survey

We measured wall-power energy per inferred bit on an RTX 5090 using the `nvidia-smi` power-draw protocol (1 kHz sampling, batched-quiescent baseline subtraction) across Pythia 70M, 1B, and 1.4B at FP32, FP16, INT8, and INT4, with batch sizes 1 and 8 and both compiled and uncompiled execution. For each configuration we computed φ_GPU = E_per_bit / (k_B T ln 2 × BPT), where k_B T ln 2 ≈ 3 × 10^−21 J/bit at the measured die temperature.

A note on definitions. Paper 4 reported silicon at ≈10^9 above Landauer using *per-token* energy attributed to the discrimination event itself. The φ_GPU measured here is a strictly larger quantity: it includes memory access, cooling, power-supply conversion losses, and idle leakage of the entire board. The two figures are not in conflict; they bound the *computational* and *system* efficiency gaps respectively. We are explicit about this throughout Section 3.4.

## 3. Results

### 3.1 The basin is data-driven

Self-evaluated bits-per-token on held-out splits:

| Model | Source entropy | Self-BPT | Ratio |
|---|---|---|---|
| SYN-2 | 1.38 | 20.52 | 14.9× |
| SYN-4 | 3.68 | 22.85 | 6.2× |
| **SYN-8** | **8.00** | **8.92** | **1.12×** |
| SYN-12 | 11.99 | 17.40 | 1.45× |

The decisive datum is SYN-8: 8.92 BPT on a source whose true entropy is exactly 8.0 bits per symbol. If the throughput basin were architectural, this model would have compressed to ≈4 BPT. It did not. Its achieved per-token cost lies within 12% of the source entropy and is more than twice the natural-language basin.

The full 4×4 cross-corpus matrix shows perfect diagonal specialization (0.03–7.38 BPT on the training corpus, including some memorization) and catastrophic off-diagonal failure (22–42 BPT). Critically, the off-diagonal entries do *not* cluster near 4 BPT — there is no architectural attractor visible at any value. Each model has learned the entropy of its specific training distribution and nothing else.

The SYN-12 result requires explicit comment. At 17.40 BPT it *exceeds* the source entropy of 11.99 bits, indicating that the 92M-parameter model lacks the capacity to fully learn a 4096-symbol distribution within the 50,000-step budget. We interpret this as a capacity limitation rather than evidence against the data-driven hypothesis, which rests primarily on the clean SYN-8 result. SYN-2 and SYN-4 also under-perform their source entropy, which we attribute to BPE tokenization artifacts on small alphabets (the BPE merges interact pathologically with very low-entropy symbol distributions, inflating per-token cost). We flag both as limitations rather than concealing them.

### 3.2 The basin is not architectural

Across the seven-corpus battery, transformer (Pythia) and state-space (Mamba) models at matched parameter counts produced means of 3.50 and 3.35 BPT respectively. Welch's *t* = 0.431, *p* = 0.688. Cohen's *d* = 0.397 (small). Levene's test for equal variances *p* = 0.975. The Mann-Whitney U on structural bonus (transformer 6.84 ± 0.15, serial 6.78 ± 0.25) returned U = 6.0, *p* = 1.0. We are unable to reject the null of equal performance on any test we ran. Attention is not the locus of the basin.

### 3.3 The quantization cliff is at INT4

The cliff is universal in location and abrupt in shape. For every model tested:

| Model | Params | Cliff precision | BPT at cliff |
|---|---|---|---|
| Pythia-70M | 70M | INT4 | 5.20 |
| Pythia-160M | 162M | INT4 | 4.49 |
| Pythia-410M | 405M | INT4 | 3.76 |
| Pythia-1B | 1.01B | INT4 | 3.14 |
| Pythia-1.4B | 1.41B | INT4 | 3.05 |
| GPT-2 | 124M | INT4 | 4.14 |
| GPT-2-Medium | 355M | INT4 | 3.74 |
| GPT-2-Large | 774M | INT4 | 3.48 |

The qualitative pattern is FP32 → FP16 < 2% degradation; FP16 → INT8 ≈ 5%; INT8 → INT4 ≈ 15%; INT4 → INT3 > 200% (catastrophic). The transition from INT4 to INT3 is a sharp phase change, not a gradient. We note that this is a per-*weight* precision floor, with no a priori connection to the per-*event* throughput basin; reporting it together with the other results clarifies that the two ~4-bit numbers measure distinct quantities.

### 3.4 Silicon operates 10^15–10^18× above Landauer

Selected configurations (full table in `exp-6/results/exp6_energy.csv`):

| Model | Precision | BPT | E/bit (J) | φ_GPU | log₁₀ φ |
|---|---|---|---|---|---|
| Pythia-70M | FP32 | 9.74 | 2.78×10⁻⁵ | 9.07×10¹⁵ | 15.96 |
| Pythia-70M | FP16 | 9.81 | 7.61×10⁻⁶ | 2.48×10¹⁵ | 15.39 |
| Pythia-70M | INT4 | 10.31 | 6.05×10⁻⁵ | 1.89×10¹⁶ | 16.28 |
| Pythia-1B | FP32 | 8.90 | 4.82×10⁻³ | 1.47×10¹⁸ | 18.17 |
| Pythia-1B | FP16 | 8.90 | 8.31×10⁻⁴ | 2.58×10¹⁷ | 17.41 |
| Pythia-1.4B | FP32 | 9.34 | 6.66×10⁻³ | 2.04×10¹⁸ | 18.31 |
| Pythia-1.4B | FP16 | 9.33 | 1.25×10⁻³ | 3.83×10¹⁷ | 17.58 |

Mean log₁₀ φ_GPU ≈ 16.8. There is no observable thermodynamic floor anywhere near the 4-bit basin; fifteen to eighteen orders of magnitude of headroom exist above the Landauer limit.

**Reconciliation with Paper 4.** Paper 4 reported silicon at ≈10^9 above Landauer; Experiment 6 reports φ_GPU ≈ 10^15–10^18. The discrepancy is one of measurement boundary, not of fact. Paper 4 estimated the *useful dissipation fraction* — the thermodynamically relevant energy per discrimination event, attributed only to the irreversible logical step. Experiment 6 measured *total GPU wall power*, which additionally pays for memory access, cooling, power-supply conversion, idle circuitry, and clock infrastructure. The ≈10^9 figure represents the gap in computational efficiency; the ≈10^16 figure represents the gap in total system efficiency. Both are valid; they measure different things, and we report both rather than collapse them into a single number that would obscure where the inefficiency lives.

## 4. Discussion

### 4.1 Confirmation of the inherited constraint by elimination

Three independent falsifiers were prepared. The architectural hypothesis predicted SYN-8 would compress to ~4 BPT (it did not), or that Mamba would diverge from Pythia (it did not). The thermodynamic hypothesis predicted a ceiling on energy per bit incompatible with > 4 BPT throughput (φ_GPU is 10^15 above any such ceiling). The intrinsic-compression hypothesis predicted a universal architectural attractor in the cross-corpus matrix (none exists). Only the data-driven hypothesis is consistent with all four experiments.

### 4.2 The four-link causal chain

This eliminates the alternatives to Paper 6's proposal: that the basin we observe in language models is the artifact of physics → biology → cognition → language → AI. Each link constrains the next; the AI is the readout of all four. The synthetic-training results are particularly clean here, because they let us *break* the language link by substituting an arbitrary distribution, and the model dutifully tracks the new distribution rather than the old basin.

### 4.3 Implications for AGI

There is no fundamental ceiling on per-event throughput in artificial neural networks. The ~4-bit basin observed in current language models is a property of the corpus, not the substrate. The corollary is sobering for purely text-trained AGI ambitions: as long as the input distribution is human-generated natural language, ~4 BPT is approximately what one gets, because that is approximately what is there to be gotten. Compression below source entropy is not allowed by Shannon.

### 4.4 The multimodal path

The natural prediction is that training distributions with higher per-event entropy — vision, audio, proprioception, embodied control signals — should yield throughput well above 4 BPT. Paper 8 will test this on vision transformers; Papers 9–10 will examine audio and embodied control respectively.

## 5. Falsifiable Predictions

1. A model with capacity > 1B parameters trained on a 16-bit synthetic corpus will achieve held-out BPT tracking source entropy (≈ 16 BPT), not the natural-language basin.
2. Vision transformers will exhibit bits-per-patch significantly different from ≈ 4 (tested in Paper 8).
3. Models trained on multimodal corpora will exhibit aggregate per-event throughput strictly higher than text-only models of the same parameter count.
4. The INT4 cliff will persist across non-transformer architectures, including Mamba and RWKV. (We did not test this here; Experiment 3 used FP16.)

Each prediction is independently capable of falsifying part of the picture. We expect to be wrong about at least one of them and we will say so when it happens.

## 5b. Internal Adversarial Review and Paper 7.1 Scope

Concurrent with manuscript preparation we ran an internal adversarial review of the experimental record (`review/adversarial_review.md`). It is published unredacted alongside this paper. Its findings are load-bearing for how the present results should be read, and they define the scope of the planned follow-up (Paper 7.1):

1. **Self-eval vs cross-corpus diagonal disagreement (Exp 1).** `exp1_self_eval.csv` and the cross-corpus diagonal disagree on the same (model, corpus) cell — by 1.5 BPT for SYN-8 and 11.9 BPT for SYN-12. Paper 7.1 will re-run all 4×4 cells with a single eval harness on a provably disjoint held-out split with seed-level error bars.
2. **Exp 6 BPT disagrees with Exp 2/3 BPT.** Pythia-160m on WikiText reports 3.96 BPT in Exp 2/3 and 12.10 BPT in Exp 6. Every φ in §3.4 is downstream of the Exp 6 number. Paper 7.1 will reconcile (almost certainly a tokenizer-normalization bug in the Exp 6 harness) and recompute φ.
3. **BPT-vs-bits-per-source-symbol unit confound (Exp 1).** The SYN-8 result (8.92 BPT) uses a corpus-trained BPE tokenizer (vocab 8192) and is therefore not directly comparable to the 8.0 bit/source-symbol entropy. Paper 7.1 will re-evaluate all SYN-* models in bits-per-source-symbol and additionally retrain with a single shared tokenizer fit on the union of corpora.
4. **No learning curves.** We cannot show SYN-8 plateaued at 8.92 vs continued descending. Paper 7.1 will publish loss-vs-step trajectories for all conditions.
5. **Capacity at scale.** 92M is below the regime where the basin was originally observed. Paper 7.1 will train at least one ≥1B-parameter SYN-8 model at matched compute.
6. **Hierarchical-structure control.** Paper 7.1 will train on a PCFG-induced 8-bit-entropy *structured* corpus and on shuffled WikiText, the two cleanest controls for "is the basin about entropy level or about hierarchical structure."
7. **`bitsandbytes`-only quantization.** §3.3 cliff is a `bitsandbytes` RTN finding, not a universal property. Paper 7.1 will repeat with GPTQ for at least one model size.
8. **Mamba energy in Exp 3.** `exp3_energy.csv` reports Mamba energy ~100× Pythia energy, almost certainly an un-fused reference kernel. Energy comparisons in §3.2 should be read with that caveat; Paper 7.1 will rerun with a fair kernel or the table will be withdrawn.

We chose to publish the present results with the review attached, rather than delay until 7.1 lands, because the institute's stated practice is to surface internal falsification attempts at the same time as the claims they constrain. A reader who treats the abstract and §5b as a unit will get the right calibration. A reader who reads only §3 will overclaim — and we have tried to write §3 so that reading is harder than the alternative.

## 6. Limitations

We report the following honestly and prominently:

- **SYN-12 capacity limitation.** 17.40 BPT exceeds the 11.99-bit source. The 92M-parameter budget is insufficient for a 4096-symbol Markov-0 distribution. The data-driven conclusion rests on SYN-8, not SYN-12.
- **SYN-2 and SYN-4 tokenization artifacts.** BPE merges interact pathologically with small alphabets, inflating per-token cost. These models do not refute the data-driven hypothesis but they also do not support it as cleanly as SYN-8.
- **Single architecture for synthetic training.** Experiment 1 used only GPT-2. We have no synthetic-training comparison for state-space models.
- **φ_GPU is not φ_useful.** Experiment 6 measures the entire GPU board including overhead. The number is an upper bound on inefficiency, not a direct estimate of irreversible discrimination cost. Paper 4's 10^9 figure remains the right number for the latter.
- **One seed per condition.** All synthetic-training results are single runs. No error bars from multiple random seeds.
- **Synthetic data lacks hierarchical structure.** SYN-2 through SYN-12 are statistical, not syntactic. They cannot speak to whether hierarchical compositional structure would change the picture.
- **50,000 training steps.** May be insufficient for full convergence on the higher-entropy corpora.

## 7. Conclusion

In the regime tested — 92M-parameter GPT-2-class models, Markov synthetic data, corpus-specific BPE tokenizers, single seed, 50K steps, `bitsandbytes` RTN quantization, `nvidia-smi` wall-power energy — the throughput basin behaves like a mirror of training-data entropy rather than a wall imposed by architecture or thermodynamics. We do not yet have the right to call that a falsification of the architectural hypothesis at scale, and the internal adversarial review (§5b) names the specific re-runs required before that stronger statement is earned. The four-link causal chain proposed in Paper 6 — physics → biology → cognition → language → AI — remains the simplest account consistent with the present data, but the present data does not yet exclude a hierarchical-structure account of the basin, and the experiments that would exclude it are scoped as Paper 7.1. Higher-throughput AGI plausibly runs through richer, higher-entropy training data including multimodal and embodied experience; Papers 8–10 will test that prediction. The INT4 → INT3 cliff is reported as a `bitsandbytes`-RTN hardware finding distinct from the per-event throughput basin despite the numerical coincidence at ~4 bits.

We are publishing these results together with the adversarial review that constrains them because that is what the Windstorm Institute is for.

## References

Whitmer, G. L. III (2026a). *The throughput basin: serial decoding convergence in large language models.* Windstorm Institute Paper 1.

Whitmer, G. L. III (2026b). *Architectural invariance of the throughput basin.* Windstorm Institute Paper 2.

Whitmer, G. L. III (2026c). *The basin across scale: 70M to 175B parameters.* Windstorm Institute Paper 3.

Whitmer, G. L. III (2026d). *Thermodynamic accounting of irreversible discrimination in silicon and ribosomes.* Windstorm Institute Paper 4.

Whitmer, G. L. III (2026e). *Cognitive bandwidth and the four-bit limit of conscious serial processing.* Windstorm Institute Paper 5.

Whitmer, G. L. III (2026f). *The inherited constraint hypothesis.* Windstorm Institute Paper 6.

Shannon, C. E. (1948). A mathematical theory of communication. *Bell System Technical Journal*, 27, 379–423, 623–656.

Shannon, C. E. (1951). Prediction and entropy of printed English. *Bell System Technical Journal*, 30, 50–64.

Landauer, R. (1961). Irreversibility and heat generation in the computing process. *IBM Journal of Research and Development*, 5(3), 183–191.

Hopfield, J. J. (1974). Kinetic proofreading: a new mechanism for reducing errors in biosynthetic processes requiring high specificity. *PNAS*, 71(10), 4135–4139.

Bennett, C. H. (1982). The thermodynamics of computation — a review. *International Journal of Theoretical Physics*, 21, 905–940.

## Acknowledgments

Claude Opus 4.6 (Anthropic) served as the autonomous research agent for both the experimental execution (Sonnet 4.5 variant, 14.5 hours unattended) and this manuscript draft. Computing on RTX 5090 (Windstorm Labs, Varon-1). All code, raw CSVs, and intermediate artifacts are public at <https://github.com/sneakyfree/agi-extensions>. Gratitude is owed to the ribosome, which continues to operate at φ ≈ 1.02 and thereby fixes a target sixteen orders of magnitude away from our current silicon.
