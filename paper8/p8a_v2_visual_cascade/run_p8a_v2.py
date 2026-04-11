#!/usr/bin/env python3
"""
Paper 8 Experiment A v2: Visual Shuffling Cascade (High-Resolution)

v1 failed because CIFAR-100 at 32×32 has too little spatial structure
for 8×8 patches (only 16 patches per image). This version uses:
- STL-10 at 96×96 (real photographs, 3× the resolution)
- 16×16 patches (36 patches per image — enough hierarchy)
- Larger model (512d, 8 layers, 8 heads, ~33M params)
- Longer training (50K steps)
- Proper convergence verification

The goal is to measure f(visual_structure) — how many bits per pixel
does spatial hierarchy contribute to a model's predictive power.
"""

import os, sys, time, math, csv, json, subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader

OUTDIR = "/home/user1-gpu/agi-extensions/paper8/p8a_v2_visual_cascade"
REPO = "/home/user1-gpu/agi-extensions"
SEEDS = [42, 137]

# Higher-res config
PATCH_SIZE = 16         # 16×16 patches (like ViT-Base)
IMG_SIZE = 96           # STL-10 native resolution
PATCHES_PER_IMG = (IMG_SIZE // PATCH_SIZE) ** 2  # 36 patches per image
PATCH_DIM = PATCH_SIZE * PATCH_SIZE * 3          # 768 raw values per patch
EMBED_DIM = 512
N_LAYERS = 8
N_HEADS = 8
TOTAL_STEPS = 50000
BATCH_SIZE = 32
LR = 3e-4
EVAL_EVERY = 5000
LOG_EVERY = 500
WARMUP_STEPS = 2000

SHUFFLE_LEVELS = [
    'original',
    'quadrant_shuffled',
    'block_4x4_shuffled',   # shuffle 4×4 grid of patches (coarse)
    'patch_shuffled',       # shuffle individual 16×16 patches
    'row_shuffled',
    'pixel_shuffled',
]

def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(f"{OUTDIR}/run.log", "a") as f:
        f.write(line + "\n")

class PatchDataset(Dataset):
    def __init__(self, images, patch_size, shuffle_mode='original'):
        self.images = images.astype(np.float32) / 255.0
        self.patch_size = patch_size
        self.shuffle_mode = shuffle_mode
        self.h_patches = images.shape[1] // patch_size
        self.w_patches = images.shape[2] // patch_size
        self.n_patches = self.h_patches * self.w_patches

    def __len__(self):
        return len(self.images)

    def _shuffle_image(self, img):
        if self.shuffle_mode == 'original':
            return img

        h, w = img.shape[:2]
        ps = self.patch_size

        if self.shuffle_mode == 'quadrant_shuffled':
            mh, mw = h // 2, w // 2
            quads = [img[:mh, :mw].copy(), img[:mh, mw:].copy(),
                     img[mh:, :mw].copy(), img[mh:, mw:].copy()]
            perm = np.random.permutation(4)
            result = np.zeros_like(img)
            positions = [(0, 0), (0, mw), (mh, 0), (mh, mw)]
            for i, p in enumerate(perm):
                r, c = positions[i]
                result[r:r+mh, c:c+mw] = quads[p]
            return result

        elif self.shuffle_mode == 'block_4x4_shuffled':
            # Shuffle 4×4 macro-blocks (each containing multiple patches)
            block_h = h // 4
            block_w = w // 4
            blocks = []
            for bi in range(4):
                for bj in range(4):
                    blocks.append(img[bi*block_h:(bi+1)*block_h,
                                     bj*block_w:(bj+1)*block_w].copy())
            np.random.shuffle(blocks)
            result = np.zeros_like(img)
            idx = 0
            for bi in range(4):
                for bj in range(4):
                    result[bi*block_h:(bi+1)*block_h,
                           bj*block_w:(bj+1)*block_w] = blocks[idx]
                    idx += 1
            return result

        elif self.shuffle_mode == 'patch_shuffled':
            patches = []
            for i in range(self.h_patches):
                for j in range(self.w_patches):
                    patches.append(img[i*ps:(i+1)*ps, j*ps:(j+1)*ps].copy())
            np.random.shuffle(patches)
            result = np.zeros_like(img)
            idx = 0
            for i in range(self.h_patches):
                for j in range(self.w_patches):
                    result[i*ps:(i+1)*ps, j*ps:(j+1)*ps] = patches[idx]
                    idx += 1
            return result

        elif self.shuffle_mode == 'row_shuffled':
            return img[np.random.permutation(h)]

        elif self.shuffle_mode == 'pixel_shuffled':
            flat = img.reshape(-1, img.shape[-1])
            return flat[np.random.permutation(len(flat))].reshape(img.shape)

        return img

    def __getitem__(self, idx):
        img = self._shuffle_image(self.images[idx].copy())
        ps = self.patch_size
        patches = []
        for i in range(self.h_patches):
            for j in range(self.w_patches):
                patch = img[i*ps:(i+1)*ps, j*ps:(j+1)*ps]
                patches.append(patch.flatten())
        return torch.tensor(np.array(patches), dtype=torch.float32)

class PatchPredictor(nn.Module):
    def __init__(self, patch_dim, embed_dim, n_layers, n_heads, n_patches):
        super().__init__()
        self.patch_dim = patch_dim
        self.embed_dim = embed_dim
        self.input_proj = nn.Linear(patch_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, embed_dim) * 0.02)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=embed_dim * 4, dropout=0.1,
            activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(embed_dim, patch_dim)
        self.register_buffer('causal_mask',
            nn.Transformer.generate_square_subsequent_mask(n_patches))
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, patches):
        B, N, D = patches.shape
        x = self.input_proj(patches) + self.pos_embed[:, :N]
        memory = torch.zeros(B, 1, self.embed_dim, device=x.device)
        mask = self.causal_mask[:N, :N].to(x.device)
        x = self.transformer(x, memory, tgt_mask=mask)
        return self.output_proj(x[:, :-1])

def compute_metrics(model, dataloader, device, pixel_var):
    """Compute bits per pixel via rate-distortion + raw MSE."""
    model.eval()
    total_mse = 0.0
    total_dims = 0

    with torch.no_grad():
        for patches in dataloader:
            patches = patches.to(device)
            B, N, D = patches.shape
            with autocast():
                predictions = model(patches)
                targets = patches[:, 1:]
                mse = F.mse_loss(predictions.float(), targets.float(), reduction='sum')
            total_mse += mse.item()
            total_dims += B * (N - 1) * D

    avg_mse = total_mse / total_dims

    # Rate-distortion: R(D) = max(0, 0.5 * log2(var / MSE))
    if avg_mse < pixel_var and avg_mse > 0:
        bits_per_dim = 0.5 * math.log2(pixel_var / avg_mse)
    else:
        bits_per_dim = 0.0

    model.train()
    return {
        'mse_per_dim': avg_mse,
        'bits_per_pixel': bits_per_dim,
        'bits_per_patch': bits_per_dim * PATCH_DIM,
    }

def train_and_evaluate(seed, train_images, eval_images, pixel_var):
    log(f"\n{'='*60}")
    log(f"Training (seed={seed}) — {IMG_SIZE}×{IMG_SIZE}, {PATCH_SIZE}×{PATCH_SIZE} patches")
    log(f"{'='*60}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_dataset = PatchDataset(train_images, PATCH_SIZE, 'original')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    model = PatchPredictor(PATCH_DIM, EMBED_DIM, N_LAYERS, N_HEADS, PATCHES_PER_IMG)
    n_params = sum(p.numel() for p in model.parameters())
    model = model.cuda()
    log(f"Model: {EMBED_DIM}d, {N_LAYERS}L, {N_HEADS}H, {n_params:,} params")
    log(f"Patches: {PATCHES_PER_IMG} per image ({PATCH_SIZE}×{PATCH_SIZE}), dim={PATCH_DIM}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    scaler = GradScaler()

    start_time = time.time()
    step = 0
    losses = []

    while step < TOTAL_STEPS:
        for patches in train_loader:
            if step >= TOTAL_STEPS:
                break
            step += 1

            if step < WARMUP_STEPS:
                lr = LR * step / WARMUP_STEPS
            else:
                progress = (step - WARMUP_STEPS) / (TOTAL_STEPS - WARMUP_STEPS)
                lr = LR * 0.5 * (1 + math.cos(math.pi * progress))
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            patches = patches.cuda()
            optimizer.zero_grad()
            with autocast():
                predictions = model(patches)
                targets = patches[:, 1:]
                loss = F.mse_loss(predictions, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            losses.append((step, loss.item()))

            if step % LOG_EVERY == 0:
                log(f"  step {step}/{TOTAL_STEPS}, loss={loss.item():.6f}, lr={lr:.6f}")

            if step % EVAL_EVERY == 0:
                eval_ds = PatchDataset(eval_images[:1000], PATCH_SIZE, 'original')
                eval_dl = DataLoader(eval_ds, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)
                m = compute_metrics(model, eval_dl, 'cuda', pixel_var)
                log(f"  EVAL step {step}: MSE={m['mse_per_dim']:.6f}, bits/px={m['bits_per_pixel']:.4f}")

    elapsed = time.time() - start_time
    log(f"Training complete: {elapsed/3600:.2f}h")

    # Save training curve
    with open(f"{OUTDIR}/results/curve_seed{seed}.csv", 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['step', 'loss'])
        w.writerows(losses)

    # Run full shuffling cascade
    log(f"\nShuffling cascade evaluation:")
    results = []
    for level in SHUFFLE_LEVELS:
        np.random.seed(seed + hash(level) % 10000)
        ds = PatchDataset(eval_images, PATCH_SIZE, level)
        dl = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)
        m = compute_metrics(model, dl, 'cuda', pixel_var)

        results.append({
            'shuffle_level': level, 'seed': seed,
            'mse_per_dim': m['mse_per_dim'],
            'bits_per_pixel': m['bits_per_pixel'],
            'bits_per_patch': m['bits_per_patch'],
            'n_params': n_params, 'train_time_hrs': elapsed/3600,
            'img_size': IMG_SIZE, 'patch_size': PATCH_SIZE,
        })
        log(f"  {level:25s}: MSE={m['mse_per_dim']:.6f}, bits/px={m['bits_per_pixel']:.4f}")

    # Compute structural bonus
    orig_bpp = [r for r in results if r['shuffle_level'] == 'original'][0]['bits_per_pixel']
    pixel_bpp = [r for r in results if r['shuffle_level'] == 'pixel_shuffled'][0]['bits_per_pixel']
    bonus = orig_bpp - pixel_bpp

    for r in results:
        r['structural_bonus'] = bonus
        r['delta_from_original'] = r['bits_per_pixel'] - orig_bpp

    log(f"\n  STRUCTURAL BONUS: {bonus:.4f} bits/pixel")

    del model
    torch.cuda.empty_cache()

    return results, bonus

def main():
    os.makedirs(f"{OUTDIR}/results", exist_ok=True)
    os.makedirs(f"{OUTDIR}/plots", exist_ok=True)

    log("="*60)
    log("P8-A v2: VISUAL SHUFFLING CASCADE (STL-10, 96×96)")
    log("="*60)

    # Load STL-10
    log("\nLoading STL-10...")
    from torchvision.datasets import STL10

    train_set = STL10(root='/tmp/stl10', split='train', download=True)
    test_set = STL10(root='/tmp/stl10', split='test', download=True)

    train_images = np.array([np.array(img) for img, _ in train_set])
    eval_images = np.array([np.array(img) for img, _ in test_set])

    log(f"Train: {train_images.shape}, Eval: {eval_images.shape}")

    # Global pixel variance
    pixel_var = np.var(train_images.astype(np.float32) / 255.0)
    log(f"Global pixel variance: {pixel_var:.6f}")

    all_results = []
    bonuses = []

    for seed in SEEDS:
        results, bonus = train_and_evaluate(seed, train_images, eval_images, pixel_var)
        all_results.extend(results)
        bonuses.append(bonus)

        # Save incrementally
        csv_path = f"{OUTDIR}/results/cascade.csv"
        file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
        with open(csv_path, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            if not file_exists:
                w.writeheader()
            w.writerows(results)

        # Git commit per seed
        try:
            subprocess.run(['git', 'add', 'paper8/p8a_v2_visual_cascade/'], cwd=REPO, capture_output=True)
            subprocess.run(['git', 'commit', '-m',
                f'Paper 8 P8-A v2: visual cascade seed={seed}, bonus={bonus:.4f}\n\n'
                f'Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>'],
                cwd=REPO, capture_output=True)
            subprocess.run(['git', 'push'], cwd=REPO, capture_output=True)
        except: pass

    # Plots
    log("\nGenerating plots...")
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import pandas as pd

        df = pd.DataFrame(all_results)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        labels = ['Original', 'Quadrants', 'Blocks 4×4', 'Patches', 'Rows', 'Pixels']

        for seed in SEEDS:
            sd = df[df['seed'] == seed]
            bpp = [sd[sd['shuffle_level'] == l]['bits_per_pixel'].values[0] for l in SHUFFLE_LEVELS]
            ax1.plot(range(len(SHUFFLE_LEVELS)), bpp, 'o-', label=f'Seed {seed}', markersize=8, linewidth=2)

        ax1.set_xticks(range(len(SHUFFLE_LEVELS)))
        ax1.set_xticklabels(labels, fontsize=9, rotation=15)
        ax1.set_ylabel('Bits per pixel', fontsize=11)
        ax1.set_title(f'Visual Shuffling Cascade — STL-10 (96×96)\nStructural bonus: {np.mean(bonuses):.3f} bpp', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Delta plot
        for seed in SEEDS:
            sd = df[df['seed'] == seed]
            deltas = [sd[sd['shuffle_level'] == l]['delta_from_original'].values[0] for l in SHUFFLE_LEVELS]
            ax2.bar([x + (0.35 if seed == SEEDS[1] else 0) for x in range(len(SHUFFLE_LEVELS))],
                   deltas, width=0.35, label=f'Seed {seed}', alpha=0.8)

        ax2.set_xticks([x + 0.175 for x in range(len(SHUFFLE_LEVELS))])
        ax2.set_xticklabels(labels, fontsize=9, rotation=15)
        ax2.set_ylabel('Δ bits/pixel from original', fontsize=11)
        ax2.set_title('Per-level structural contribution', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        fig.savefig(f"{OUTDIR}/plots/cascade_v2.png", dpi=300, bbox_inches='tight')
        fig.savefig(f"{OUTDIR}/plots/cascade_v2.pdf", bbox_inches='tight')
        plt.close(fig)
        log("Plots saved")
    except Exception as e:
        log(f"Plot error: {e}")

    # Report
    mean_bonus = np.mean(bonuses)
    report = f"""# Paper 8 P8-A v2: Visual Shuffling Cascade (High-Resolution)

## Changes from v1
- STL-10 (96×96) instead of CIFAR-100 (32×32) — 9× more pixels
- 16×16 patches (36 per image) instead of 8×8 (16 per image)
- Larger model: 512d, 8L, 8H, ~33M params (was 384d, 6L, 6H, 14M)
- 50K steps (was 30K)
- Added block_4x4_shuffled level for finer spatial decomposition

## Results

| Level | Bits/pixel (mean) | Δ from original |
|---|---|---|
"""
    for level in SHUFFLE_LEVELS:
        level_data = [r for r in all_results if r['shuffle_level'] == level]
        mean_bpp = np.mean([r['bits_per_pixel'] for r in level_data])
        delta = mean_bpp - np.mean([r['bits_per_pixel'] for r in all_results if r['shuffle_level'] == 'original'])
        report += f"| {level} | {mean_bpp:.4f} | {delta:+.4f} |\n"

    report += f"""
## Structural Bonus: {mean_bonus:.4f} bits/pixel

Language structural bonus (Paper 6): ~6.7 bits/token
PCFG-8 structural bonus (Paper 7 R6): ~5.3 bits/token
"""
    with open(f"{OUTDIR}/results/P8A_V2_REPORT.md", 'w') as f:
        f.write(report)

    # Final commit
    try:
        subprocess.run(['git', 'add', 'paper8/p8a_v2_visual_cascade/'], cwd=REPO, capture_output=True)
        subprocess.run(['git', 'commit', '-m',
            f'Paper 8 P8-A v2 complete: STL-10 cascade, bonus={mean_bonus:.4f}\n\n'
            f'Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>'],
            cwd=REPO, capture_output=True)
        subprocess.run(['git', 'push'], cwd=REPO, capture_output=True)
    except: pass

    log(f"\n{'='*60}")
    log(f"P8-A v2 COMPLETE — Structural bonus: {mean_bonus:.4f} bits/pixel")
    log(f"{'='*60}")

if __name__ == "__main__":
    main()
