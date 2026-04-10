#!/usr/bin/env python3
"""
Paper 7.1 - B4: Partial retrain of SYN-8 with multiple seeds, logging train
loss and held-out BPT / BPSS* every LOG_STEPS on all 4 corpora.

Mirrors exp-1/code/exp1_train.py hyperparameters. Re-uses the exp-1 SYN-8
tokenizer (already saved). Trains from scratch to MAX_STEPS per seed.
Saves learning curves to CSV.
"""
import os, sys, math, time, json, csv
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast,
    get_linear_schedule_with_warmup,
)

ROOT = Path('/home/user1-gpu/agi-extensions')
CORPUS_DIR = ROOT / 'exp-1' / 'corpora'
TOK_DIR = ROOT / 'exp-1' / 'tokenizers' / 'syn8'
OUT = ROOT / 'paper7.1'
OUT.mkdir(exist_ok=True, parents=True)

LOG_FILE = OUT / 'run.log'
def log(msg):
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')

CORPORA = ['syn2', 'syn4', 'syn8', 'syn12']
SEQ_LEN = 512
BATCH_SIZE = 16
LR = 3e-4
WARMUP = 1000
MAX_STEPS = int(os.environ.get('MAX_STEPS', '60000'))
LOG_STEPS = 100
EVAL_EVERY = 2000     # full BPT/BPSS* eval (expensive)
EVAL_TOKENS = 200_000 # tokens per eval corpus per eval round
SEEDS = [int(s) for s in os.environ.get('SEEDS', '42,137,2024').split(',')]

def load_tokenizer():
    return PreTrainedTokenizerFast.from_pretrained(str(TOK_DIR))

def load_corpus_token_ids(corpus_name, tokenizer):
    """Load a corpus and tokenize ALL of it once (cached in RAM)."""
    p = CORPUS_DIR / f'{corpus_name}.txt'
    with open(p, 'r') as f:
        text = f.read()
    ids = tokenizer(text, add_special_tokens=False)['input_ids']
    return text, np.array(ids, dtype=np.int32)

def make_train_batches(train_ids, seq_len, batch_size, rng):
    """Infinite generator: yields random-offset batches from train_ids."""
    n = len(train_ids)
    max_start = n - seq_len - 1
    while True:
        starts = rng.integers(0, max_start, size=batch_size)
        batch = np.stack([train_ids[s:s+seq_len] for s in starts])
        yield torch.from_numpy(batch).long()

@torch.no_grad()
def eval_bpt_bpss(model, ids_eval, raw_text_eval, seq_len, max_tokens, device):
    """Evaluate on held-out token stream. Uses non-overlapping windows."""
    model.eval()
    ln2 = math.log(2)
    ids = ids_eval[:max_tokens]
    n_windows = len(ids) // seq_len
    total_bits = 0.0
    total_scored = 0
    for w in range(0, n_windows, 16):
        batch = []
        for k in range(w, min(w+16, n_windows)):
            batch.append(ids[k*seq_len:(k+1)*seq_len])
        batch_t = torch.from_numpy(np.stack(batch)).long().to(device)
        out = model(batch_t, labels=batch_t)
        scored = batch_t.shape[0] * (seq_len - 1)
        total_bits += out.loss.item() * scored / ln2
        total_scored += scored
    # chars corresponding to the tokens we actually evaluated
    # raw_text_eval is the decoded span for these exact tokens (provided)
    total_chars = len(raw_text_eval)
    bpt = total_bits / max(1, total_scored)
    bpss = total_bits / max(1, total_chars)
    model.train()
    return bpt, bpss, total_bits, total_scored, total_chars

def main():
    log("=== B4 partial retrain starting ===")
    log(f"MAX_STEPS={MAX_STEPS} SEEDS={SEEDS}")
    tokenizer = load_tokenizer()
    vocab_size = tokenizer.vocab_size
    log(f"Vocab size: {vocab_size}")

    # Pre-tokenize SYN-8 training data (we retrain SYN-8)
    log("Tokenizing SYN-8 training corpus...")
    t0 = time.time()
    syn8_text, syn8_ids = load_corpus_token_ids('syn8', tokenizer)
    split_train = int(len(syn8_ids) * 0.9)
    train_ids = syn8_ids[:split_train]
    log(f"  SYN-8: {len(syn8_ids):,} tokens, train={len(train_ids):,} ({time.time()-t0:.1f}s)")

    # Pre-tokenize held-out eval streams for all 4 corpora (last 5% of raw text)
    eval_streams = {}
    for c in CORPORA:
        p = CORPUS_DIR / f'{c}.txt'
        with open(p, 'r') as f:
            text = f.read()
        n = len(text)
        heldout_text = text[int(n*0.95):]
        probe = tokenizer(heldout_text[:200_000], add_special_tokens=False)['input_ids']
        chars_per_tok = 200_000 / max(1, len(probe))
        need = min(len(heldout_text), int(EVAL_TOKENS * chars_per_tok * 1.2))
        ids = np.array(
            tokenizer(heldout_text[:need], add_special_tokens=False)['input_ids'][:EVAL_TOKENS],
            dtype=np.int32,
        )
        # decoded span == source chars for BPSS*
        try:
            decoded = tokenizer.decode(ids.tolist(), skip_special_tokens=True)
        except Exception:
            decoded = heldout_text[:int(len(ids)*chars_per_tok)]
        eval_streams[c] = (ids, decoded)
        log(f"  eval[{c}]: tokens={len(ids)}, chars={len(decoded)}")

    device = torch.device('cuda')
    all_rows = []
    for seed in SEEDS:
        log(f"\n=== Seed {seed} ===")
        torch.manual_seed(seed)
        np.random.seed(seed)
        rng = np.random.default_rng(seed)

        config = GPT2Config(
            vocab_size=vocab_size,
            n_embd=768, n_layer=12, n_head=12, n_positions=SEQ_LEN,
            bos_token_id=tokenizer.cls_token_id,
            eos_token_id=tokenizer.sep_token_id,
        )
        model = GPT2LMHeadModel(config).to(device)
        nparams = sum(p.numel() for p in model.parameters())
        log(f"Model params: {nparams:,}")

        optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        sched = get_linear_schedule_with_warmup(optim, WARMUP, MAX_STEPS)
        scaler = torch.amp.GradScaler('cuda')

        gen = make_train_batches(train_ids, SEQ_LEN, BATCH_SIZE, rng)
        t_start = time.time()
        running_loss = 0.0
        running_n = 0
        model.train()
        for step in range(1, MAX_STEPS+1):
            batch = next(gen).to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', dtype=torch.float16):
                out = model(batch, labels=batch)
                loss = out.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            sched.step()
            running_loss += loss.item()
            running_n += 1

            if step % LOG_STEPS == 0:
                avg_loss = running_loss / running_n
                running_loss = 0.0; running_n = 0
                # Light log (no eval): append a row with NaN eval
                row = dict(model='syn8', seed=seed, step=step,
                           train_loss=f"{avg_loss:.6f}",
                           eval_corpus='', eval_loss_BPT='', eval_BPSS_star='')
                all_rows.append(row)

            if step % EVAL_EVERY == 0 or step == MAX_STEPS:
                elapsed = time.time() - t_start
                log(f"  step {step}/{MAX_STEPS} train_loss={loss.item():.4f} "
                    f"elapsed={elapsed:.0f}s ({step/elapsed:.1f} it/s)")
                for c in CORPORA:
                    ids_c, decoded_c = eval_streams[c]
                    bpt, bpss, bits, toks, chars = eval_bpt_bpss(
                        model, ids_c, decoded_c, SEQ_LEN, EVAL_TOKENS, device)
                    row = dict(model='syn8', seed=seed, step=step,
                               train_loss=f"{loss.item():.6f}",
                               eval_corpus=c,
                               eval_loss_BPT=f"{bpt:.6f}",
                               eval_BPSS_star=f"{bpss:.6f}")
                    all_rows.append(row)
                    log(f"    eval[{c}] BPT={bpt:.4f} BPSS*={bpss:.4f}")
                # flush intermediate CSV
                with open(OUT/'b4_learning_curves.csv','w',newline='') as f:
                    w = csv.DictWriter(f, fieldnames=['model','seed','step','train_loss','eval_loss_BPT','eval_BPSS_star','eval_corpus'])
                    w.writeheader()
                    for r in all_rows:
                        w.writerow(r)
        del model, optim, sched
        torch.cuda.empty_cache()
    log("B4 done.")

if __name__ == '__main__':
    main()
