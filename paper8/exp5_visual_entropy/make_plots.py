#!/usr/bin/env python3
"""Generate plots for exp5 visual entropy measurement."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/user1-gpu/agi-extensions/paper8/exp5_visual_entropy"
df = pd.read_csv(f"{ROOT}/results/visual_entropy.csv")

# ---- Plot 1: visual_entropy_comparison.png ----
fig, ax = plt.subplots(figsize=(12,7))
datasets = df["dataset"].tolist()
x = np.arange(len(datasets))
w = 0.22

H_cond = df["H_conditional"].values
H_fgz = df["H_filtered_gzip_mean"].values
H_png = df["H_png_mean"].values

ax.bar(x - w, H_cond, w, label="H_conditional (left neighbor)", color="#2196F3")
ax.bar(x,     H_fgz,  w, label="H_filtered_gzip (sub + deflate)", color="#FF9800")
ax.bar(x + w, H_png,  w, label="H_png (Paeth + deflate)", color="#4CAF50")

# Try to show WebP if available
if "H_jpeg_lossless_mean" in df.columns:
    webp = pd.to_numeric(df["H_jpeg_lossless_mean"], errors="coerce").values
    if not np.all(np.isnan(webp)):
        ax.bar(x + 2*w, webp, w, label="H_webp_lossless", color="#9C27B0")

ax.axhline(8.0, color="red", ls="--", lw=1, label="8.0 bpp (theoretical max)")
ax.axhline(4.16, color="orange", ls="--", lw=1, label="4.16 bpp (language throughput basin)")
ax.axhline(1.0, color="gray", ls="--", lw=1, label="1.0 bpp (aggressive lossy)")

ax.set_xticks(x)
ax.set_xticklabels(datasets, rotation=25, ha="right")
ax.set_ylabel("Bits per pixel")
ax.set_ylim(0, 10)
ax.legend(fontsize=8, loc="upper right")

# Headline: find tightest estimator for natural datasets
nat = df[~df["dataset"].isin(["RandomNoise","ConstantColor"])]
webp_nat = pd.to_numeric(nat["H_jpeg_lossless_mean"], errors="coerce")
if webp_nat.notna().any():
    tightest = webp_nat.mean()
    est_name = "WebP lossless"
else:
    tightest = nat["H_png_mean"].mean()
    est_name = "PNG"
ax.set_title(f"Natural images: ~{tightest:.1f} bpp under {est_name} compression\n"
             f"(language throughput basin = 4.16 bpp)", fontsize=13)
plt.tight_layout()
fig.savefig(f"{ROOT}/plots/visual_entropy_comparison.png", dpi=150)
print("Saved visual_entropy_comparison.png")

# ---- Plot 2: spatial_entropy_cascade.png ----
fig2, ax2 = plt.subplots(figsize=(9,6))
blocks = ["H_block_1x1","H_block_2x2","H_block_4x4","H_block_8x8"]
block_labels = ["1x1","2x2","4x4","8x8"]

for name, style, color in [("CIFAR-100","-o","#2196F3"),("RandomNoise","--s","#F44336"),("ConstantColor","--^","#9E9E9E")]:
    row = df[df["dataset"]==name]
    if row.empty: continue
    vals = [float(row[b].values[0]) for b in blocks]
    ax2.plot(block_labels, vals, style, color=color, label=name, lw=2, markersize=8)

ax2.axhline(8.0, color="red", ls=":", lw=1, alpha=0.5)
ax2.axhline(4.16, color="orange", ls=":", lw=1, alpha=0.5)
ax2.set_xlabel("Block size")
ax2.set_ylabel("Bits per pixel (block entropy / block pixels)")
ax2.set_title("Spatial entropy cascade: entropy per pixel vs block size\n"
              "(block entropy at k≥4 is biased low due to histogram sparsity)")
ax2.legend()
ax2.set_ylim(-0.5, 10)
plt.tight_layout()
fig2.savefig(f"{ROOT}/plots/spatial_entropy_cascade.png", dpi=150)
print("Saved spatial_entropy_cascade.png")
print("Done")
