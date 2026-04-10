#!/usr/bin/env python3
"""
Paper 7.1 - B1: Unified evaluation harness.

Resolves the 1.54-BPT discrepancy between exp-1 self-eval (SYN-8 = 8.92) and
cross-corpus diagonal (SYN-8 = 7.38).

Root cause (confirmed by reading exp1_evaluate.py):
  - self_evaluation: takes text[0.9*len:][:100000] i.e. the FIRST 100k chars of
    the held-out slice, then compute_bpt truncates to max_length=512 TOKENS.
    So it only ever scores a single 512-token window -> noisy single-seq CE.
  - cross_corpus_evaluation: takes text[:100000] i.e. the FIRST 100k chars of
    the WHOLE corpus. This is INSIDE the training split (train = first 90%),
    so the diagonal number (7.38 for SYN-8) is contaminated by training data
    leakage. Also truncated to 512 tokens.

Unified harness fixes both:
  - Use last 5% of raw text (well inside the trained test split of last 10%,
    and disjoint from training's first 90%).
  - Tokenize with MODEL'S OWN tokenizer.
  - Evaluate over MANY non-overlapping 512-token windows (we cap total tokens
    evaluated at MAX_EVAL_TOKENS to keep runtime bounded), summing CE over
    tokens to produce a stable BPT.
  - Report BPT, total_bits, total_tokens, total_source_chars, and
    BPSS* = total_bits / total_source_chars.
"""
import os, sys, time, math, json, csv, traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

ROOT = Path('/home/user1-gpu/agi-extensions')
CORPUS_DIR = ROOT / 'exp-1' / 'corpora'
MODEL_DIR = ROOT / 'exp-1' / 'models'
OUT = ROOT / 'paper7.1'
OUT.mkdir(exist_ok=True, parents=True)

LOG = OUT / 'run.log'
def log(msg):
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')

CORPORA = ['syn2', 'syn4', 'syn8', 'syn12']
SEQ_LEN = 512
MAX_EVAL_TOKENS = 500_000   # ~1000 windows; plenty for stable BPT
HELDOUT_FRAC = 0.05         # last 5% of raw text

def load_model(name):
    path = MODEL_DIR / name / 'final'
    tok = PreTrainedTokenizerFast.from_pretrained(str(path))
    model = GPT2LMHeadModel.from_pretrained(str(path), torch_dtype=torch.float16).cuda()
    model.eval()
    return model, tok

def get_heldout_text(corpus_name):
    p = CORPUS_DIR / f'{corpus_name}.txt'
    with open(p, 'r') as f:
        text = f.read()
    n = len(text)
    split_idx = int(n * (1 - HELDOUT_FRAC))
    # Training was first 90% -> heldout (last 5%) is strictly disjoint with a
    # 5% safety margin between train and eval.
    heldout = text[split_idx:]
    return heldout, n, split_idx

@torch.no_grad()
def unified_eval(model, tokenizer, raw_text, max_tokens=MAX_EVAL_TOKENS):
    """Tokenize raw_text, chunk to SEQ_LEN windows, compute total CE in bits.

    Returns dict with BPT, total_bits, total_tokens, total_source_chars, BPSS*
    """
    # Determine how many chars we actually need to reach max_tokens budget.
    # Probe tokenization density on first 200k chars to estimate.
    probe_chars = min(len(raw_text), 200_000)
    probe_ids = tokenizer(raw_text[:probe_chars], add_special_tokens=False)['input_ids']
    if len(probe_ids) == 0:
        return None
    chars_per_tok = probe_chars / max(1, len(probe_ids))
    need_chars = int(max_tokens * chars_per_tok * 1.1)
    need_chars = min(need_chars, len(raw_text))
    used_text = raw_text[:need_chars]

    ids = tokenizer(used_text, add_special_tokens=False)['input_ids']
    ids = ids[:max_tokens]
    total_tokens_input = len(ids)
    if total_tokens_input < SEQ_LEN + 1:
        return None

    # Figure out the source-char footprint of exactly those tokens.
    # We decode them back to get a deterministic char count corresponding to
    # the tokens we actually evaluate. (BPE decode may re-add spaces; for
    # these whitespace-pretok BPE tokenizers this is fine, and it guarantees
    # BPSS* uses the same span the bits are computed over.)
    try:
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        total_source_chars = len(decoded)
    except Exception:
        total_source_chars = int(total_tokens_input * chars_per_tok)

    # Evaluate via non-overlapping SEQ_LEN windows; the loss skips the first
    # token of each window (standard HF shift), so bits = sum(loss_nats)*(SEQ_LEN-1)/ln2.
    device = next(model.parameters()).device
    ln2 = math.log(2)

    total_bits = 0.0
    total_scored_tokens = 0

    n_windows = total_tokens_input // SEQ_LEN
    # Batch windows for speed
    batch_size = 16
    for bstart in range(0, n_windows, batch_size):
        bend = min(bstart + batch_size, n_windows)
        batch = []
        for w in range(bstart, bend):
            start = w * SEQ_LEN
            batch.append(ids[start:start + SEQ_LEN])
        batch_t = torch.tensor(batch, dtype=torch.long, device=device)
        out = model(batch_t, labels=batch_t)
        # out.loss is mean over (B*(SEQ_LEN-1)) tokens
        scored = batch_t.shape[0] * (SEQ_LEN - 1)
        nats = out.loss.item() * scored
        total_bits += nats / ln2
        total_scored_tokens += scored

    bpt = total_bits / total_scored_tokens
    bpss_star = total_bits / max(1, total_source_chars)
    return dict(
        BPT=bpt,
        total_bits=total_bits,
        total_tokens=total_scored_tokens,
        total_source_chars=total_source_chars,
        BPSS_star=bpss_star,
    )

def main():
    log("=== B1 unified harness starting ===")
    log(f"GPU: {torch.cuda.get_device_name(0)}")

    rows = []
    # Preload held-out texts
    heldouts = {}
    for c in CORPORA:
        h, n, split = get_heldout_text(c)
        heldouts[c] = h
        log(f"{c}: total_chars={n:,}, heldout_start={split:,}, heldout_chars={len(h):,}")

    for model_name in CORPORA:
        try:
            log(f"Loading model {model_name}")
            model, tok = load_model(model_name)
        except Exception as e:
            log(f"FAILED loading {model_name}: {e}")
            traceback.print_exc()
            continue
        for corpus in CORPORA:
            t0 = time.time()
            try:
                res = unified_eval(model, tok, heldouts[corpus])
                if res is None:
                    log(f"  {model_name} x {corpus}: insufficient data")
                    continue
                row = dict(
                    model=model_name,
                    eval_corpus=corpus,
                    seed=42,
                    BPT=f"{res['BPT']:.6f}",
                    BPSS_star=f"{res['BPSS_star']:.6f}",
                    total_bits=f"{res['total_bits']:.3f}",
                    total_tokens=res['total_tokens'],
                    total_source_chars=res['total_source_chars'],
                    split_method='last_5pct_raw_text_disjoint',
                    timestamp=datetime.now().isoformat(timespec='seconds'),
                )
                rows.append(row)
                log(f"  {model_name} x {corpus}: BPT={res['BPT']:.4f} BPSS*={res['BPSS_star']:.4f} "
                    f"bits={res['total_bits']:.1f} toks={res['total_tokens']} "
                    f"chars={res['total_source_chars']} ({time.time()-t0:.1f}s)")
            except Exception as e:
                log(f"  {model_name} x {corpus} FAILED: {e}")
                traceback.print_exc()
        del model, tok
        torch.cuda.empty_cache()

    # Write CSV
    out_csv = OUT / 'b1_unified_matrix.csv'
    cols = ['model','eval_corpus','seed','BPT','BPSS_star','total_bits','total_tokens','total_source_chars','split_method','timestamp']
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    log(f"Wrote {out_csv} ({len(rows)} rows)")

if __name__ == '__main__':
    main()
