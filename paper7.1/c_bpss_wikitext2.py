#!/usr/bin/env python3
"""
Paper 7.1 - Experiment C: Full BPSS* rollout on WikiText-2.

Evaluates a set of HF pre-trained LMs on WikiText-2 test split and reports
both BPT and BPSS* = total_bits / total_source_chars.

Protocol:
  - Load WikiText-2-raw-v1 test split, join with '\n' to get the raw text.
  - total_source_chars = len(raw text) BEFORE tokenization. This is what
    BPSS* must normalize against.
  - Tokenize with model's own tokenizer, evaluate CE in non-overlapping
    max_len-token windows, sum loss -> total_bits.
  - Runs each model; catches exceptions per-model so one failure doesn't
    kill the rest.
"""
import os, sys, math, time, csv, traceback, gc
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

OUT = Path('/home/user1-gpu/agi-extensions/paper7.1')
LOG = OUT / 'run.log'
def log(msg):
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')

MODELS = [
    ('EleutherAI/pythia-160m', 'transformer', 162_000_000),
    ('EleutherAI/pythia-410m', 'transformer', 405_000_000),
    ('EleutherAI/pythia-1.4b', 'transformer', 1_400_000_000),
    ('gpt2-medium',            'transformer', 355_000_000),
    ('state-spaces/mamba-130m-hf', 'mamba', 130_000_000),
    ('state-spaces/mamba-370m-hf', 'mamba', 370_000_000),
    ('state-spaces/mamba-1.4b-hf', 'mamba', 1_400_000_000),
]

MAX_LEN = 1024

@torch.no_grad()
def eval_model(name, arch, nparams, raw_text, device):
    log(f"Loading {name}")
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16).to(device)
    model.eval()

    ids = tok(raw_text, return_tensors='pt', add_special_tokens=False).input_ids[0]
    total_tokens_input = ids.shape[0]
    log(f"  tokens={total_tokens_input:,} raw_chars={len(raw_text):,}")

    ln2 = math.log(2)
    total_bits = 0.0
    total_scored = 0
    # Determine a window length safely
    wl = MAX_LEN
    # mamba doesn't have attention window, but 1024 still fine
    batch_size = 2 if nparams >= 1_000_000_000 else 4
    n_windows = total_tokens_input // wl
    for w in range(0, n_windows, batch_size):
        chunk = []
        for k in range(w, min(w+batch_size, n_windows)):
            chunk.append(ids[k*wl:(k+1)*wl])
        batch = torch.stack(chunk).to(device)
        out = model(batch, labels=batch)
        scored = batch.shape[0] * (wl - 1)
        total_bits += out.loss.item() * scored / ln2
        total_scored += scored

    bpt = total_bits / max(1, total_scored)
    bpss = total_bits / max(1, len(raw_text))
    log(f"  {name}: BPT={bpt:.4f} BPSS*={bpss:.4f} bits={total_bits:.1f}")

    del model, tok
    gc.collect()
    torch.cuda.empty_cache()
    return dict(
        model=name, architecture=arch, params=nparams,
        BPT=f"{bpt:.6f}", BPSS_star=f"{bpss:.6f}",
        total_bits=f"{total_bits:.3f}",
        total_tokens=total_scored,
        total_source_chars=len(raw_text),
    )

def main():
    log("=== Experiment C: BPSS* on WikiText-2 ===")
    log("Loading WikiText-2 test split...")
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    raw_text = '\n'.join(ds['text'])
    log(f"WT2 test raw chars: {len(raw_text):,}")

    device = torch.device('cuda')
    rows = []
    for name, arch, params in MODELS:
        try:
            rows.append(eval_model(name, arch, params, raw_text, device))
        except Exception as e:
            log(f"FAILED {name}: {e}")
            traceback.print_exc()

    out_csv = OUT / 'bpss_exp3_wikitext2.csv'
    cols = ['model','architecture','params','BPT','BPSS_star','total_bits','total_tokens','total_source_chars']
    with open(out_csv,'w',newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows: w.writerow(r)
    log(f"Wrote {out_csv} ({len(rows)} rows)")

if __name__ == '__main__':
    main()
