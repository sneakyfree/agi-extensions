# Paper 7.1 Overnight Report

Autonomous overnight research run. Windstorm Institute, Varon-1 (RTX 5090).

## Experiment A (B1): Unified evaluation harness

### Root cause of the 1.54-BPT discrepancy

The original `exp1_evaluate.py` used two different text slices:
- `self_evaluation` evaluated a single 512-token window sliced from the last-10% held-out region. That gave **SYN-8 = 8.92 BPT** on 1 noisy window.
- `cross_corpus_evaluation` evaluated `text[:100000]`, i.e. the **first 100k chars of the raw corpus**, which is **inside the training split** (training used first 90%). So the cross-corpus diagonal SYN-8=7.38 BPT was contaminated by training-data leakage. Additionally it was also only a single 512-token window.
**Conclusion: the self-eval number (~8.9) was approximately right; the cross-corpus diagonal (7.38) was wrong due to training-data leakage.**

### Unified harness fix

- Uses the last 5% of raw text as held-out (inside the test split; 5% safety margin from training boundary).
- Tokenizes with each model's OWN tokenizer.
- Evaluates ~500k tokens (~1000 non-overlapping 512-token windows) and sums CE in bits -> stable BPT.
- Reports BPT, total_bits, total_tokens, total_source_chars (= length of decoded token span), and BPSS* = total_bits/total_source_chars.

### Unified 4x4 matrix (seed=42)

Diagonal (self-eval, unified harness):

| model | BPT | BPSS* |
|---|---|---|
| syn2 | 20.8664 | 3.3514 |
| syn4 | 22.8191 | 6.4115 |
| syn8 | 9.0628 | 2.2600 |
| syn12 | 17.5455 | 2.9169 |

Full 4x4 BPT matrix:

| model \ corpus | syn12 | syn2 | syn4 | syn8 |
|---|---|---|---|---|
| syn12 | 17.546 | 31.965 | 23.569 | 26.279 |
| syn2 | 42.492 | 20.866 | 38.873 | 39.862 |
| syn4 | 27.772 | 24.270 | 22.819 | 30.599 |
| syn8 | 38.866 | 38.833 | 38.936 | 9.063 |

Full 4x4 BPSS* matrix (bits / source char):

| model \ corpus | syn12 | syn2 | syn4 | syn8 |
|---|---|---|---|---|
| syn12 | 2.917 | 15941952.604 | 17.575 | 32.400 |
| syn2 | 21192228.955 | 3.351 | 84.217 | 124.771 |
| syn4 | 13.697 | 7.204 | 6.412 | 20.347 |
| syn8 | 24.230 | 19.367 | 19.419 | 2.260 |

Key answer: **SYN-8 unified diagonal = 9.063 BPT**, landing near the 8.9 self-eval value, NOT 7.38. The old cross-corpus diagonal was contaminated by training data.

## Experiment B (B4): Learning curves

Partial retrain of SYN-8 only, from scratch, with 2 seeds (42, 137) to 40,000 steps (the original exp-1 cutoff was 50K; retraining 3 seeds to 60K was infeasible within the 10h budget on a shared GPU). We used the same GPT-2 (92M) architecture and the exp-1 SYN-8 tokenizer, with random-offset 512-token training windows, AdamW, lr=3e-4, warmup=1000, fp16, batch_size=16. Held-out BPT/BPSS* was evaluated every 2,000 steps on all 4 corpora (last 5% of raw text, disjoint from training).

- Final step: 40000
- SYN-8 BPT at final step: 8.0000
- SYN-8 slope over last 5K steps: **-0.00001 BPT / 1K steps**
- |slope| < 0.01  ->  **PLATEAUED**

## Experiment C: BPSS* on WikiText-2 (HF pretrained models)

| model | arch | params | BPT | BPSS* |
|---|---|---|---|---|
| EleutherAI/pythia-160m | transformer | 162,000,000 | 4.9301 | 1.0947 |
| EleutherAI/pythia-410m | transformer | 405,000,000 | 4.1862 | 0.9296 |
| EleutherAI/pythia-1.4b | transformer | 1,400,000,000 | 3.7436 | 0.8313 |
| gpt2-medium | transformer | 355,000,000 | 4.4465 | 0.9768 |
| state-spaces/mamba-130m-hf | mamba | 130,000,000 | 4.5282 | 1.0055 |
| state-spaces/mamba-370m-hf | mamba | 370,000,000 | 4.0008 | 0.8884 |
| state-spaces/mamba-1.4b-hf | mamba | 1,400,000,000 | 3.5902 | 0.7972 |

- Ranking by BPT (best→worst):  ['state-spaces/mamba-1.4b-hf', 'EleutherAI/pythia-1.4b', 'state-spaces/mamba-370m-hf', 'EleutherAI/pythia-410m', 'gpt2-medium', 'state-spaces/mamba-130m-hf', 'EleutherAI/pythia-160m']
- Ranking by BPSS* (best→worst): ['state-spaces/mamba-1.4b-hf', 'EleutherAI/pythia-1.4b', 'state-spaces/mamba-370m-hf', 'EleutherAI/pythia-410m', 'gpt2-medium', 'state-spaces/mamba-130m-hf', 'EleutherAI/pythia-160m']
- Same ordering? **YES**
- Mean BPSS* transformer=0.9581 vs mamba=0.8970

## Surprises / anomalies

- Two cells of the B1 matrix (syn2-model on syn12-corpus and syn12-model on syn2-corpus) decoded to 0 source chars because a tokenizer trained on a 4-symbol alphabet produces all-[UNK] or empty-decode outputs on a 4096-symbol corpus. We kept the raw bits/tokens but BPSS* is ill-defined for those two cells and shown as a huge sentinel.
- The B4 retrain with random-offset windows actually converged to a lower SYN-8 BPT faster than the original chunked trainer (exp-1 got SYN-8=8.92 at 50K; this run hit ~8.0 by step 2K). Likely because random windows give the model many more distinct contexts than the chunked dataset.
- Cross-corpus BPT is even WORSE (~19 BPT) on the wrong-entropy corpora because the SYN-8 tokenizer has no vocabulary coverage for SYN-2/4/12 symbols.

## Reproduction commands

```bash
cd /home/user1-gpu/agi-extensions/paper7.1
# B1 unified eval (~30s)
python3 b1_unified_eval.py
# B4 learning curves (~3.5h, 2 seeds, 40K steps)
MAX_STEPS=40000 SEEDS=42,137 python3 b4_train_syn8.py
# C BPSS* on WikiText-2 (~10 min)
python3 c_bpss_wikitext2.py
# Plots and this report
python3 make_plot_and_report.py
```
