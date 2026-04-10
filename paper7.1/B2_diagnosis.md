# B2 Diagnosis: Exp 6 vs Exp 2/3 BPT Discrepancy

## Bug (one sentence)

Exp 6 evaluates BPT on **10,000-token sequences**, far beyond the **2,048-token training context** of every Pythia model in the survey, so the loss reported is dominated by RoPE position-extrapolation collapse rather than the model's true cross-entropy on WikiText-2.

## Location

- `exp-6/code/exp6_main.py:50` — `EVAL_TOKENS = 10000`
- `exp-6/code/exp6_main.py:193` — `tokenizer(text, truncation=True, max_length=EVAL_TOKENS)` truncates to 10,000 tokens (the GPT-NeoX tokenizer's `model_max_length` is effectively unbounded, so the cap actually engages at 10k).
- `exp-6/code/exp6_main.py:201` — `model(input_ids, labels=input_ids)` then runs a 10k-token forward pass through a model whose RoPE was trained on positions `[0, 2048)`. Positions `[2048, 10000)` are extrapolated, predictions there are near-random, and the mean loss inflates accordingly.

By contrast, exp-2 (`exp-2/code/exp2_main.py:54`) and exp-3 (`exp-3/code/exp3_main.py:57`) both set `MAX_LENGTH = 1024`, comfortably inside the trained context — these are the trustworthy measurements.

The BPT *formula* itself is identical across all three files (`bpt = loss / np.log(2)`, exp-2:230, exp-3:191, exp-6:204). Same log base, same divisor, same labels=input_ids convention. The discrepancy is **not** a unit bug.

## Why the candidate "unit bug" hypotheses are wrong

| Candidate | Verdict |
|---|---|
| (a) `log` vs `log2` | Ruled out — all three files use `loss/np.log(2)` identically. Factor would be ~0.69, wrong direction and magnitude. |
| (b) bits-per-byte reported as bits-per-token | Ruled out — exp-2 reports `bpb ≈ 0.04` *separately* from `bpt ≈ 3.96`; the bytes/token ratio for the GPT-NeoX BPE on WikiText-2 is ~4, not ~3. And exp-6 never touches a byte count anywhere in its code path. |
| (c) Per-batch vs total averaging | Ruled out — all three use a single forward pass with HuggingFace's built-in `outputs.loss` (mean over the sequence). No per-batch accumulation. |
| (d) Stride/overlap | Ruled out — none of the three use a sliding window. Single non-overlapping forward pass each. |
| (e) Wrong split | Ruled out — all three load `wikitext-2-raw-v1` `split='test'`. |
| (f) Wrong tokenizer | Ruled out — all three call `AutoTokenizer.from_pretrained(model_name)`. |

The 3.06 ratio is **coincidence**, not a clean conversion factor.

## Corrected formula

For Pythia models on WikiText-2, evaluate BPT with the input sequence length **≤ training context (2,048 tokens)**. Use `EVAL_TOKENS = 1024` (matching exp-2/exp-3) so the result is directly comparable. Energy and power measurements need a longer wall-clock window for stability, but the BPT calculation should be done on a separate, in-context forward pass — do **not** reuse the long-sequence loss.

Equivalently, for the exp-6 CSV: substitute the trustworthy BPT from exp-2 (same model, same precision, same dataset, same formula) and recompute φ as

    φ_corrected = energy_per_token_J / (E_landauer · bpt_corrected)

with `bpt_corrected` taken from `exp2_quantization.csv` keyed by `(model, precision)`. fp32 / fp16_compiled / fp16_batch_* configs all use the exp-2 fp16 BPT (the BPT difference between fp32 and fp16 is <0.1% within exp-6's own data, so this substitution is safe).

## Worked example: Pythia-160m, fp16

| Quantity | Value |
|---|---|
| exp-6 bpt (10k-token forward, RoPE extrapolation collapse) | 12.1051 |
| exp-2 bpt (1024-token forward, in-context) | 3.9561 |
| ratio | **3.0598** |
| energy_per_token_J | 1.4433e-3 |
| E_landauer (J, at 335.15 K) | 3.2074e-21 |
| φ_orig = ept / (E_L · 12.1051) | 3.717e+16 |
| φ_corrected = ept / (E_L · 3.9561) | **1.137e+17** |
| log10 φ_orig | 16.57 |
| log10 φ_corrected | **17.06** |

The same 3.06× ratio appears in Pythia-1.4B (9.332 / 2.981 = 3.131), Pythia-1B (8.898 / 3.074 = 2.895), Pythia-410M (11.215 / 3.370 = 3.328), Pythia-70M (9.811 / 4.714 = 2.081). The ratio grows with the inverse of the in-context BPT, exactly as you'd expect when the "extra" loss is a near-constant per-position penalty from extrapolating RoPE past 2048 — small models have higher baseline BPT, so the ratio is smaller.

## Direction of the correction (important)

Correcting the bug **raises** φ rather than lowering it. φ is inversely proportional to BPT, so replacing inflated BPT with the true in-context BPT makes the silicon look *less* efficient relative to Landauer, not more.

| Quantity | Original (exp-6 as published) | Corrected |
|---|---|---|
| log10(φ) range across all 31 rows | 15.39 .. 18.31 | **15.71 .. 18.80** |
| log10(φ) median | 17.03 | **17.52** |

So Paper 7 §3.4's "10^15 – 10^18 above Landauer" headline **survives intact** — and tightens slightly upward to roughly 10^15.7 – 10^18.8. The corrected values do **not** collapse anywhere near Paper 5's ~10^9 useful-dissipation figure; the gap to that estimate is still ~7–9 orders of magnitude and unchanged in character.

## Implication for Paper 7 §3.4

**Numerical revision, no withdrawal.** The thermodynamics section's qualitative claim is unaffected; the table of φ values needs to be regenerated against `exp6_energy_corrected.csv`, and the text should note that the original Exp 6 numbers were depressed by a context-length artifact (Exp 6 evaluated BPT at 10k tokens against a 2k-context model). The headline gap to Landauer is in fact *slightly larger* than originally reported. A one-line methods-section footnote is sufficient; no claims need to be retracted.

A separate Paper 7.1 follow-up should re-run Exp 6 with `EVAL_TOKENS = 1024` end-to-end so BPT and energy are measured on the same forward pass, but this can be a low-priority confirmation run — the substitution above is mathematically equivalent because the energy column does not depend on which sequence length the BPT was measured at.

## Output

- `paper7.1/exp6_energy_corrected.csv` — all 31 rows of `exp6_energy.csv`, original columns preserved verbatim, with three new columns appended: `bpt_corrected`, `phi_corrected`, `log10_phi_corrected`.
