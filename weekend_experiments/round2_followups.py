#!/usr/bin/env python3
"""
Round 2 Follow-up Experiments
Priority order:
1. P8-E4b: Audio with real LJ Speech (GPU, ~3 hrs)
2. P8-E3: ImageNet cascade at 224×224 (GPU, ~6 hrs)
3. P7-PCFG-depth: PCFG depth sweep for f() curve (GPU, ~8 hrs)
4. P7-tau: Re-measure τ in bits-per-source-byte (CPU, ~1 hr)
"""

import os, sys, time, math, csv, subprocess, traceback, io, gzip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader

REPO = "/home/user1-gpu/agi-extensions"
BASE = "/home/user1-gpu/agi-extensions/weekend_experiments"

def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(f"{BASE}/round2.log", "a") as f:
        f.write(line + "\n")

def git_commit_push(msg, paths):
    try:
        for p in paths:
            subprocess.run(['git', 'add', p], cwd=REPO, capture_output=True)
        subprocess.run(['git', 'commit', '-m', msg + '\n\nCo-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>'],
                       cwd=REPO, capture_output=True)
        subprocess.run(['git', 'push'], cwd=REPO, capture_output=True)
    except: pass

def save_csv(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not data: return
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
        w.writeheader()
        w.writerows(data)

# =====================================================================
# 1. P8-E4b: Audio with Real LJ Speech
# =====================================================================
def run_p8_e4b():
    OUTDIR = f"{BASE}/p8_e4b_real_audio"
    os.makedirs(f"{OUTDIR}/results", exist_ok=True)
    os.makedirs(f"{OUTDIR}/plots", exist_ok=True)
    log("=" * 60)
    log("P8-E4b: AUDIO WITH REAL LJ SPEECH")
    log("=" * 60)

    import torchaudio
    import librosa

    n_mels = 128
    hop_length = 512
    sr = 22050
    seq_len = 64
    embed_dim = 256
    n_layers = 4
    n_heads = 4
    total_steps = 30000
    batch_size = 32
    seeds = [42, 137]

    # Try multiple download approaches for LJ Speech
    audio_data = {}

    # Speech: LJ Speech
    log("  Downloading LJ Speech via torchaudio...")
    try:
        os.makedirs('/tmp/ljspeech', exist_ok=True)
        dataset = torchaudio.datasets.LJSPEECH(root='/tmp/ljspeech', download=True)
        waveforms = []
        for i in range(min(1000, len(dataset))):
            waveform, sample_rate, _, _ = dataset[i]
            if sample_rate != sr:
                resampler = torchaudio.transforms.Resample(sample_rate, sr)
                waveform = resampler(waveform)
            waveforms.append(waveform.numpy().flatten())
            if i % 200 == 0:
                log(f"    Loaded {i}/{min(1000, len(dataset))} utterances")
        audio_raw = np.concatenate(waveforms)
        log(f"  LJ Speech loaded: {len(audio_raw)/sr:.0f} seconds")
        audio_data['speech_real'] = audio_raw
    except Exception as e:
        log(f"  LJ Speech failed: {e}")
        log("  Falling back to torchaudio.datasets.YESNO (tiny but real speech)")
        try:
            dataset = torchaudio.datasets.YESNO(root='/tmp/yesno', download=True)
            waveforms = []
            for i in range(len(dataset)):
                waveform, sample_rate, _ = dataset[i]
                if sample_rate != sr:
                    resampler = torchaudio.transforms.Resample(sample_rate, sr)
                    waveform = resampler(waveform)
                waveforms.append(waveform.numpy().flatten())
            audio_raw = np.concatenate(waveforms)
            log(f"  YESNO loaded: {len(audio_raw)/sr:.0f} seconds")
            audio_data['speech_real'] = audio_raw
        except Exception as e2:
            log(f"  YESNO also failed: {e2}")
            log("  Generating high-quality synthetic speech (formants + modulation)")
            t = np.linspace(0, 600, sr * 600)
            # Better synthetic: multiple formants with amplitude modulation
            f0 = 120 + 30 * np.sin(2 * np.pi * 0.5 * t)  # varying pitch
            audio_raw = (
                0.4 * np.sin(2 * np.pi * f0 * t) *  # F0
                (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)) +  # amplitude modulation (syllables)
                0.2 * np.sin(2 * np.pi * 2.5 * f0 * t) +  # F1
                0.1 * np.sin(2 * np.pi * 4 * f0 * t) +  # F2
                0.03 * np.random.randn(len(t))  # breath noise
            ).astype(np.float32)
            audio_data['speech_synthetic'] = audio_raw

    # Music: Generate realistic piano with overtones
    log("  Generating piano music...")
    duration = 600
    t_full = np.linspace(0, duration, sr * duration)
    audio_music = np.zeros(len(t_full), dtype=np.float32)
    rng = np.random.RandomState(42)
    midi_notes = [60, 62, 64, 65, 67, 69, 71, 72, 74, 76, 77, 79]  # C major scale + octave

    for note_start in range(0, duration * sr, sr // 4):  # quarter note = 250ms
        note_end = min(note_start + sr, len(t_full))
        midi = midi_notes[rng.randint(len(midi_notes))]
        freq = 440 * 2 ** ((midi - 69) / 12)
        amp = rng.uniform(0.05, 0.15)
        seg_t = t_full[note_start:note_end] - t_full[note_start]
        envelope = np.exp(-seg_t * 3)  # piano decay
        # Fundamental + 3 harmonics (piano timbre)
        tone = (amp * np.sin(2 * np.pi * freq * seg_t) +
                amp * 0.5 * np.sin(2 * np.pi * 2 * freq * seg_t) +
                amp * 0.25 * np.sin(2 * np.pi * 3 * freq * seg_t) +
                amp * 0.125 * np.sin(2 * np.pi * 4 * freq * seg_t))
        audio_music[note_start:note_end] += tone * envelope

    audio_data['music'] = audio_music

    # Noise
    audio_data['noise'] = np.random.randn(sr * 600).astype(np.float32) * 0.3

    # Convert all to mel spectrograms and train
    from weekend_experiments.phase3_audio_and_controlled_entropy import AudioFrameDataset, AudioPredictor

    all_results = []

    for source_name, audio_raw in audio_data.items():
        log(f"\n  Processing {source_name} ({len(audio_raw)/sr:.0f} seconds)...")

        mel = librosa.feature.melspectrogram(y=audio_raw, sr=sr, n_mels=n_mels,
                                             hop_length=hop_length, n_fft=2048)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_frames = mel_db.T.astype(np.float32)
        mel_frames = (mel_frames - mel_frames.mean()) / (mel_frames.std() + 1e-8)

        log(f"    Mel frames: {mel_frames.shape}")

        split = int(len(mel_frames) * 0.9)
        train_frames = mel_frames[:split]
        eval_frames = mel_frames[split:]
        frame_var = np.var(eval_frames)

        for seed in seeds:
            log(f"    Training {source_name} seed={seed}...")
            torch.manual_seed(seed)
            np.random.seed(seed)

            train_ds = AudioFrameDataset(train_frames, seq_len)
            train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                 num_workers=2, pin_memory=True, drop_last=True)

            model = AudioPredictor(n_mels, embed_dim, n_layers, n_heads, seq_len).cuda()
            n_params = sum(p.numel() for p in model.parameters())
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
            scaler = GradScaler()

            step = 0
            start_time = time.time()
            while step < total_steps:
                for batch in train_dl:
                    if step >= total_steps: break
                    step += 1
                    lr = 3e-4 * min(step / 1000, 0.5 * (1 + math.cos(math.pi * max(0, step-1000)/(total_steps-1000))))
                    for pg in optimizer.param_groups: pg['lr'] = lr

                    batch = batch.cuda()
                    optimizer.zero_grad()
                    with autocast():
                        preds = model(batch)
                        loss = F.mse_loss(preds, batch[:, 1:])
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()

                    if step % 5000 == 0:
                        log(f"      step {step}/{total_steps}, loss={loss.item():.6f}")

            elapsed = time.time() - start_time

            # Evaluate
            model.eval()
            eval_ds = AudioFrameDataset(eval_frames, seq_len)
            eval_dl = DataLoader(eval_ds, batch_size=batch_size, num_workers=2, pin_memory=True)

            total_mse = 0.0
            total_dims = 0
            with torch.no_grad():
                for batch in eval_dl:
                    batch = batch.cuda()
                    with autocast():
                        preds = model(batch)
                        targets = batch[:, 1:]
                        mse = F.mse_loss(preds.float(), targets.float(), reduction='sum')
                    total_mse += mse.item()
                    total_dims += targets.numel()

            avg_mse = total_mse / total_dims
            bits_per_dim = max(0, 0.5 * math.log2(frame_var / avg_mse)) if avg_mse < frame_var and avg_mse > 0 else 0

            # Shuffling cascade
            cascade = []
            for mode in ['original', 'segment_shuffled', 'frame_shuffled']:
                if mode == 'original':
                    test_f = eval_frames
                elif mode == 'segment_shuffled':
                    seg = len(eval_frames) // 20
                    segs = [eval_frames[i:i+seg] for i in range(0, len(eval_frames)-seg, seg)]
                    np.random.shuffle(segs)
                    test_f = np.concatenate(segs)
                else:
                    test_f = eval_frames[np.random.permutation(len(eval_frames))]

                ds = AudioFrameDataset(test_f, seq_len)
                dl = DataLoader(ds, batch_size=batch_size, num_workers=2, pin_memory=True)
                s_mse, s_dims = 0.0, 0
                with torch.no_grad():
                    for b in dl:
                        b = b.cuda()
                        with autocast():
                            p = model(b)
                            m = F.mse_loss(p.float(), b[:, 1:].float(), reduction='sum')
                        s_mse += m.item()
                        s_dims += b[:, 1:].numel()
                s_avg = s_mse / s_dims
                s_bpd = max(0, 0.5 * math.log2(frame_var / s_avg)) if s_avg < frame_var and s_avg > 0 else 0
                cascade.append({'source': source_name, 'seed': seed, 'mode': mode,
                               'mse': s_avg, 'bits_per_dim': s_bpd})

            bonus = cascade[0]['bits_per_dim'] - cascade[-1]['bits_per_dim']

            all_results.append({
                'source': source_name, 'seed': seed,
                'bits_per_dim': bits_per_dim, 'bits_per_frame': bits_per_dim * n_mels,
                'structural_bonus': bonus, 'mse': avg_mse, 'frame_var': frame_var,
                'n_frames': len(mel_frames), 'duration_sec': len(audio_raw)/sr,
                'n_params': n_params, 'train_time_hrs': elapsed/3600,
            })
            log(f"      {source_name} seed={seed}: bits/dim={bits_per_dim:.4f}, bonus={bonus:.4f}")

            save_csv(cascade, f"{OUTDIR}/results/cascade_{source_name}_seed{seed}.csv")
            del model, optimizer, scaler
            torch.cuda.empty_cache()

        git_commit_push(f"P8-E4b: {source_name} audio complete", ['weekend_experiments/p8_e4b_real_audio/'])

    save_csv(all_results, f"{OUTDIR}/results/audio_throughput_v2.csv")

    report = "# P8-E4b: Audio with Real/Improved Sources\n\n"
    report += "| Source | Duration | Bits/mel_dim (mean) | Structural bonus |\n|---|---|---|---|\n"
    for src in audio_data.keys():
        src_r = [r for r in all_results if r['source'] == src]
        if src_r:
            report += f"| {src} | {src_r[0]['duration_sec']:.0f}s | {np.mean([r['bits_per_dim'] for r in src_r]):.4f} | {np.mean([r['structural_bonus'] for r in src_r]):.4f} |\n"
    with open(f"{OUTDIR}/results/P8_E4B_REPORT.md", 'w') as f:
        f.write(report)

    git_commit_push("P8-E4b: real audio complete", ['weekend_experiments/p8_e4b_real_audio/'])
    log("P8-E4b COMPLETE")

# =====================================================================
# 2. P7-PCFG-DEPTH: Depth sweep for f() curve
# =====================================================================
def run_p7_pcfg_depth():
    OUTDIR = f"{BASE}/p7_pcfg_depth_sweep"
    os.makedirs(f"{OUTDIR}/results", exist_ok=True)
    os.makedirs(f"{OUTDIR}/plots", exist_ok=True)
    log("=" * 60)
    log("P7 PCFG DEPTH SWEEP: Fitting f(structural_depth)")
    log("=" * 60)

    from scipy.optimize import curve_fit

    # Generate PCFG corpora at varying recursive depths
    # All target ~8-bit entropy, varying only structure depth
    depths = [1, 2, 3, 5, 8]
    corpus_size = 20_000_000  # 20M chars each (smaller for speed)
    n_terminals = 256

    all_results = []

    for max_depth in depths:
        log(f"\n  Generating PCFG at depth={max_depth}...")

        # Build grammar with controlled depth
        rng = np.random.RandomState(42)
        corpus = []
        n_categories = max(4, max_depth * 4)

        def generate_sentence(depth_remaining):
            if depth_remaining <= 0 or rng.random() < 0.3:
                # Terminal: random byte
                return [rng.randint(0, n_terminals)]
            else:
                # Non-terminal: 2-4 children
                n_children = rng.randint(2, min(5, depth_remaining + 2))
                result = []
                for _ in range(n_children):
                    result.extend(generate_sentence(depth_remaining - 1))
                return result

        while len(corpus) < corpus_size:
            sentence = generate_sentence(max_depth)
            corpus.extend(sentence)

        corpus = np.array(corpus[:corpus_size], dtype=np.uint8)

        # Verify entropy
        from collections import Counter
        counts = Counter(corpus.tolist())
        total = sum(counts.values())
        H = -sum((c/total) * math.log2(c/total) for c in counts.values())
        log(f"    depth={max_depth}: H={H:.4f} bits, {len(corpus)} symbols, {len(counts)} unique")

        # Encode as hex text and build tokenizer
        text = ' '.join(f'x{b:02x}' for b in corpus)
        split = int(len(text) * 0.9)
        train_text = text[:split]
        eval_text = text[split:]

        from tokenizers import ByteLevelBPETokenizer
        tokenizer = ByteLevelBPETokenizer()
        chunks = [train_text[i:i+10000] for i in range(0, min(len(train_text), 5000000), 10000)]
        tokenizer.train_from_iterator(chunks, vocab_size=8192, min_frequency=2,
                                       special_tokens=["<pad>", "<eos>", "<unk>"])
        tok_path = f"{OUTDIR}/tokenizer_d{max_depth}.json"
        tokenizer.save(tok_path)

        from transformers import PreTrainedTokenizerFast
        tok = PreTrainedTokenizerFast(tokenizer_file=tok_path)

        train_ids = torch.tensor(tok.encode(train_text), dtype=torch.long)
        eval_ids = torch.tensor(tok.encode(eval_text), dtype=torch.long)

        log(f"    Train: {len(train_ids)} tokens, Eval: {len(eval_ids)} tokens")

        # Also create shuffled version
        corpus_shuf = corpus.copy()
        np.random.shuffle(corpus_shuf)
        text_shuf = ' '.join(f'x{b:02x}' for b in corpus_shuf)
        eval_text_shuf = text_shuf[split:]
        eval_ids_shuf = torch.tensor(tok.encode(eval_text_shuf), dtype=torch.long)

        # Train model
        from transformers import GPT2Config, GPT2LMHeadModel

        for seed in [42]:
            log(f"    Training depth={max_depth} seed={seed}...")
            torch.manual_seed(seed)

            config = GPT2Config(vocab_size=tok.vocab_size, n_embd=768, n_layer=12, n_head=12,
                               n_positions=512, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0)
            model = GPT2LMHeadModel(config).cuda()
            model.gradient_checkpointing_enable()
            n_params = sum(p.numel() for p in model.parameters())
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
            scaler = GradScaler()

            step = 0
            total_steps = 10000
            train_rng = np.random.default_rng(seed)
            start_time = time.time()

            while step < total_steps:
                step += 1
                lr = 3e-4 * min(step/500, 0.5*(1+math.cos(math.pi*max(0,step-500)/(total_steps-500))))
                for pg in optimizer.param_groups: pg['lr'] = lr

                start_idx = train_rng.integers(0, len(train_ids) - 513)
                x = train_ids[start_idx:start_idx+512].unsqueeze(0).cuda()
                y = train_ids[start_idx+1:start_idx+513].unsqueeze(0).cuda()

                optimizer.zero_grad()
                with autocast():
                    out = model(x)
                    loss = F.cross_entropy(out.logits.view(-1, config.vocab_size), y.view(-1))
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                if step % 2000 == 0:
                    log(f"      step {step}/{total_steps}, loss={loss.item():.4f}")

            elapsed = time.time() - start_time

            # Evaluate on original
            model.eval()
            total_loss, total_tok = 0.0, 0
            with torch.no_grad():
                for s in range(0, len(eval_ids)-512, 512):
                    x = eval_ids[s:s+512].unsqueeze(0).cuda()
                    y = eval_ids[s+1:s+513].unsqueeze(0).cuda()
                    with autocast():
                        out = model(x)
                        l = F.cross_entropy(out.logits.view(-1, config.vocab_size), y.view(-1))
                    total_loss += l.item() * 512
                    total_tok += 512
            bpt_orig = (total_loss / total_tok) / math.log(2)

            # Evaluate on shuffled
            total_loss_s, total_tok_s = 0.0, 0
            with torch.no_grad():
                for s in range(0, len(eval_ids_shuf)-512, 512):
                    x = eval_ids_shuf[s:s+512].unsqueeze(0).cuda()
                    y = eval_ids_shuf[s+1:s+513].unsqueeze(0).cuda()
                    with autocast():
                        out = model(x)
                        l = F.cross_entropy(out.logits.view(-1, config.vocab_size), y.view(-1))
                    total_loss_s += l.item() * 512
                    total_tok_s += 512
            bpt_shuf = (total_loss_s / total_tok_s) / math.log(2)

            bonus = bpt_shuf - bpt_orig

            all_results.append({
                'max_depth': max_depth, 'seed': seed,
                'source_H': H, 'bpt_original': bpt_orig, 'bpt_shuffled': bpt_shuf,
                'structural_bonus': bonus, 'n_params': n_params,
                'train_time_hrs': elapsed/3600, 'corpus_size': len(corpus),
            })
            log(f"      depth={max_depth}: BPT={bpt_orig:.4f}, shuffled={bpt_shuf:.4f}, bonus={bonus:.4f}")

            del model, optimizer, scaler
            torch.cuda.empty_cache()

        git_commit_push(f"P7 PCFG depth={max_depth}", ['weekend_experiments/p7_pcfg_depth_sweep/'])

    save_csv(all_results, f"{OUTDIR}/results/pcfg_depth_sweep.csv")

    # Fit f(depth)
    log("\n  Fitting f(depth)...")
    depths_arr = np.array([r['max_depth'] for r in all_results])
    bonus_arr = np.array([r['structural_bonus'] for r in all_results])

    fits = {}
    try:
        # Logarithmic: f(d) = a * log(1 + d)
        def f_log(d, a): return a * np.log(1 + d)
        popt, _ = curve_fit(f_log, depths_arr, bonus_arr, p0=[1.0])
        fits['logarithmic'] = {'a': popt[0], 'r2': 1 - np.sum((bonus_arr - f_log(depths_arr, *popt))**2) / np.sum((bonus_arr - bonus_arr.mean())**2)}

        # Saturating: f(d) = a * (1 - exp(-b*d))
        def f_sat(d, a, b): return a * (1 - np.exp(-b * d))
        popt2, _ = curve_fit(f_sat, depths_arr, bonus_arr, p0=[6.0, 0.5], maxfev=5000)
        fits['saturating'] = {'a': popt2[0], 'b': popt2[1], 'r2': 1 - np.sum((bonus_arr - f_sat(depths_arr, *popt2))**2) / np.sum((bonus_arr - bonus_arr.mean())**2)}

        # Power: f(d) = a * d^b
        def f_pow(d, a, b): return a * np.power(d.astype(float), b)
        popt3, _ = curve_fit(f_pow, depths_arr, bonus_arr, p0=[1.0, 0.5], maxfev=5000)
        fits['power'] = {'a': popt3[0], 'b': popt3[1], 'r2': 1 - np.sum((bonus_arr - f_pow(depths_arr, *popt3))**2) / np.sum((bonus_arr - bonus_arr.mean())**2)}

        for name, fit in fits.items():
            log(f"    {name}: R²={fit['r2']:.4f}, params={fit}")
    except Exception as e:
        log(f"  Curve fitting error: {e}")

    # Plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(depths_arr, bonus_arr, s=150, zorder=5, color='#2166ac', edgecolors='black')

        d_smooth = np.linspace(0.5, 12, 100)
        if 'logarithmic' in fits:
            ax.plot(d_smooth, f_log(d_smooth, fits['logarithmic']['a']), '--',
                   label=f"log: R²={fits['logarithmic']['r2']:.3f}")
        if 'saturating' in fits:
            ax.plot(d_smooth, f_sat(d_smooth, fits['saturating']['a'], fits['saturating']['b']), '-',
                   label=f"saturating (asymptote={fits['saturating']['a']:.2f}): R²={fits['saturating']['r2']:.3f}")
        if 'power' in fits:
            ax.plot(d_smooth, f_pow(d_smooth, fits['power']['a'], fits['power']['b']), ':',
                   label=f"power: R²={fits['power']['r2']:.3f}")

        ax.axhline(y=6.7, color='red', linestyle='--', alpha=0.5, label='Natural language bonus (~6.7)')
        ax.set_xlabel('Maximum recursive depth', fontsize=12)
        ax.set_ylabel('Structural bonus (bits)', fontsize=12)
        ax.set_title('f(structural_depth) — How structure compresses the basin', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.savefig(f"{OUTDIR}/plots/f_depth_curve.png", dpi=300, bbox_inches='tight')
        fig.savefig(f"{OUTDIR}/plots/f_depth_curve.pdf", bbox_inches='tight')
        plt.close(fig)
        log("  Plot saved")
    except Exception as e:
        log(f"  Plot error: {e}")

    report = f"# PCFG Depth Sweep: Fitting f(structural_depth)\n\n"
    report += "| Depth | Source H | BPT | Shuffled BPT | Structural Bonus |\n|---|---|---|---|---|\n"
    for r in all_results:
        report += f"| {r['max_depth']} | {r['source_H']:.3f} | {r['bpt_original']:.4f} | {r['bpt_shuffled']:.4f} | {r['structural_bonus']:.4f} |\n"
    report += "\n## Curve Fits\n\n"
    for name, fit in fits.items():
        report += f"- **{name}**: R²={fit['r2']:.4f}, {fit}\n"

    with open(f"{OUTDIR}/results/PCFG_DEPTH_REPORT.md", 'w') as f:
        f.write(report)

    git_commit_push("P7 PCFG depth sweep complete", ['weekend_experiments/p7_pcfg_depth_sweep/'])
    log("PCFG DEPTH SWEEP COMPLETE")

# =====================================================================
# 3. P7-TAU: Re-measure τ in bits-per-source-byte (CPU only)
# =====================================================================
def run_p7_tau_remeasure():
    OUTDIR = f"{BASE}/p7_tau_remeasure"
    os.makedirs(f"{OUTDIR}/results", exist_ok=True)
    log("=" * 60)
    log("P7 TAU RE-MEASUREMENT IN BITS PER SOURCE BYTE")
    log("=" * 60)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    models = [
        'EleutherAI/pythia-160m',
        'EleutherAI/pythia-410m',
        'EleutherAI/pythia-1.4b',
        'gpt2-medium',
    ]

    # Load WikiText-2
    from datasets import load_dataset
    wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    raw_text = " ".join([t for t in wiki["text"] if len(t.strip()) > 0])
    n_source_chars = len(raw_text)
    n_source_bytes = len(raw_text.encode('utf-8'))
    log(f"  WikiText-2 test: {n_source_chars} chars, {n_source_bytes} bytes")

    results = []

    for model_name in models:
        log(f"\n  Evaluating {model_name}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).cuda().eval()

            input_ids = tokenizer.encode(raw_text, return_tensors='pt').squeeze()
            n_tokens = len(input_ids)
            log(f"    {n_tokens} tokens, {n_source_bytes} bytes → {n_source_bytes/n_tokens:.2f} bytes/token")

            # Compute BPT
            total_loss = 0.0
            total_tok = 0
            with torch.no_grad():
                for start in range(0, min(len(input_ids) - 1024, 50000), 1024):
                    x = input_ids[start:start+1024].unsqueeze(0).cuda()
                    y = input_ids[start+1:start+1025].unsqueeze(0).cuda()
                    outputs = model(x)
                    loss = F.cross_entropy(outputs.logits.view(-1, outputs.logits.size(-1)), y.view(-1))
                    total_loss += loss.item() * 1024
                    total_tok += 1024

            bpt = (total_loss / total_tok) / math.log(2)
            total_bits = bpt * n_tokens
            bits_per_char = total_bits / n_source_chars
            bits_per_byte = total_bits / n_source_bytes

            results.append({
                'model': model_name,
                'n_tokens': n_tokens,
                'n_chars': n_source_chars,
                'n_bytes': n_source_bytes,
                'bytes_per_token': n_source_bytes / n_tokens,
                'BPT': bpt,
                'bits_per_char': bits_per_char,
                'bits_per_byte': bits_per_byte,
            })
            log(f"    BPT={bpt:.4f}, bits/char={bits_per_char:.4f}, bits/byte={bits_per_byte:.4f}")

            del model
            torch.cuda.empty_cache()

        except Exception as e:
            log(f"    ERROR: {e}")

    save_csv(results, f"{OUTDIR}/results/tau_remeasured.csv")

    report = "# τ Re-measurement in Bits Per Source Byte\n\n"
    report += "The original τ = 4.16 ± 0.19 was in BPT (tokenizer-dependent).\n"
    report += "This re-measures in bits per source character and bits per source byte.\n\n"
    report += "| Model | BPT | Bits/char | Bits/byte | Bytes/token |\n|---|---|---|---|---|\n"
    for r in results:
        report += f"| {r['model']} | {r['BPT']:.4f} | {r['bits_per_char']:.4f} | {r['bits_per_byte']:.4f} | {r['bytes_per_token']:.2f} |\n"

    if results:
        mean_bpc = np.mean([r['bits_per_char'] for r in results])
        mean_bpb = np.mean([r['bits_per_byte'] for r in results])
        report += f"\n**Mean bits/char: {mean_bpc:.4f}**\n"
        report += f"**Mean bits/byte: {mean_bpb:.4f}**\n"
        report += f"\nFor comparison: τ (BPT) = 4.16 ± 0.19\n"
        report += f"τ (bits/char) = {mean_bpc:.2f}\n"
        report += f"τ (bits/byte) = {mean_bpb:.2f}\n"

    with open(f"{OUTDIR}/results/TAU_REPORT.md", 'w') as f:
        f.write(report)

    git_commit_push("P7 τ re-measured in bits-per-source-byte", ['weekend_experiments/p7_tau_remeasure/'])
    log("TAU RE-MEASUREMENT COMPLETE")

# =====================================================================
# MAIN
# =====================================================================
def main():
    log("=" * 60)
    log("ROUND 2 FOLLOW-UP EXPERIMENTS")
    log("=" * 60)

    # 1. Audio with real data
    try:
        run_p8_e4b()
    except Exception as e:
        log(f"P8-E4b FAILED: {e}")
        traceback.print_exc()

    # 2. PCFG depth sweep
    try:
        run_p7_pcfg_depth()
    except Exception as e:
        log(f"PCFG depth FAILED: {e}")
        traceback.print_exc()

    # 3. τ re-measurement
    try:
        run_p7_tau_remeasure()
    except Exception as e:
        log(f"τ remeasure FAILED: {e}")
        traceback.print_exc()

    log("\n" + "=" * 60)
    log("ROUND 2 COMPLETE")
    log("=" * 60)

    git_commit_push("Round 2 follow-up experiments complete", ['weekend_experiments/'])

if __name__ == "__main__":
    main()
