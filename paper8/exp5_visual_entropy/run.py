#!/usr/bin/env python3
"""Paper 8 exp 5: visual source entropy via multiple estimators. CPU only."""
import os, sys, time, gzip, zlib, io, csv, math
from multiprocessing import Pool
import numpy as np
from PIL import Image

ROOT = "/home/user1-gpu/agi-extensions/paper8/exp5_visual_entropy"
RES = f"{ROOT}/results"
PLT = f"{ROOT}/plots"
os.makedirs(RES, exist_ok=True); os.makedirs(PLT, exist_ok=True)
DATA = "/tmp/datasets"
os.makedirs(DATA, exist_ok=True)

CSV_PATH = f"{RES}/visual_entropy.csv"
FIELDS = ["dataset","n_images_used","image_height","image_width","image_channels",
          "H_pixel","H_pixel_R","H_pixel_G","H_pixel_B","H_conditional",
          "H_block_1x1","H_block_2x2","H_block_4x4","H_block_8x8",
          "H_gzip_mean","H_gzip_std","H_png_mean","H_png_std",
          "H_filtered_gzip_mean","H_filtered_gzip_std","H_jpeg_lossless_mean",
          "elapsed_seconds","notes"]

def write_header():
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH,"w",newline="") as f:
            csv.DictWriter(f, FIELDS).writeheader()

def append_row(row):
    with open(CSV_PATH,"a",newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writerow({k: row.get(k,"") for k in FIELDS})

def shannon(counts):
    counts = counts[counts>0].astype(np.float64)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())

def H_marginal_per_channel(images):
    # images: (N,H,W,C) uint8
    Hs = []
    for c in range(images.shape[-1]):
        counts = np.bincount(images[...,c].ravel(), minlength=256)
        Hs.append(shannon(counts))
    return Hs

def H_conditional_left(images):
    # for each channel, joint hist of (left, curr) over horizontally adjacent pairs
    Hs=[]
    for c in range(images.shape[-1]):
        ch = images[...,c]  # (N,H,W)
        prev = ch[:,:,:-1].ravel().astype(np.int64)
        curr = ch[:,:,1:].ravel().astype(np.int64)
        idx = prev*256 + curr
        joint = np.bincount(idx, minlength=256*256).reshape(256,256)
        H_joint = shannon(joint.ravel())
        H_prev = shannon(joint.sum(axis=1))
        Hs.append(H_joint - H_prev)
    return Hs

def H_block(images, k):
    # non-overlapping k x k blocks across all channels jointly
    N,H,W,C = images.shape
    Hc = (H//k)*k; Wc=(W//k)*k
    imgs = images[:,:Hc,:Wc,:]
    # reshape into blocks
    blocks = imgs.reshape(N, Hc//k, k, Wc//k, k, C).transpose(0,1,3,2,4,5).reshape(-1, k*k*C)
    # hash via bytes
    if k==1:
        # equivalent to per-pixel triplet; for k=1 use joint over channels? spec says k=1 == H_pixel reference
        # interpret as mean marginal
        return float(np.mean(H_marginal_per_channel(images)))
    # count via dict
    from collections import Counter
    cnt = Counter()
    # convert to bytes rows
    bview = blocks.tobytes()
    rowlen = blocks.shape[1]
    n = blocks.shape[0]
    for i in range(n):
        cnt[bview[i*rowlen:(i+1)*rowlen]] += 1
    arr = np.array(list(cnt.values()), dtype=np.float64)
    H = shannon(arr)
    return H / (k*k*C)

def gzip_one(img):
    raw = img.tobytes()
    comp = zlib.compress(raw, 9)
    return len(comp)*8 / img.size

def png_one(img):
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG", optimize=True)
    return len(buf.getvalue())*8 / img.size

def filtered_gzip_one(img):
    # sub filter: pixel - left neighbor mod 256
    a = img.astype(np.int16)
    diff = a.copy()
    diff[:,1:,:] = (a[:,1:,:] - a[:,:-1,:]) % 256
    raw = diff.astype(np.uint8).tobytes()
    return len(zlib.compress(raw,9))*8 / img.size

def webp_one(img):
    try:
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="WEBP", lossless=True, quality=100)
        return len(buf.getvalue())*8 / img.size
    except Exception:
        return None

def parallel_map(fn, items, workers=8):
    with Pool(workers) as p:
        return p.map(fn, items, chunksize=64)

def measure(name, images, sample_cap=5000, notes=""):
    t0 = time.time()
    print(f"[{name}] shape={images.shape} dtype={images.dtype}", flush=True)
    N,H,W,C = images.shape
    # marginal & conditional on full set
    Hpc = H_marginal_per_channel(images)
    Hcond = H_conditional_left(images)
    # block at k=1,2,4,8
    Hblk = {}
    for k in [1,2,4,8]:
        if k > min(H,W): Hblk[k] = ""
        else: Hblk[k] = H_block(images, k)
    # sample for compression
    if N > sample_cap:
        idx = np.random.RandomState(0).choice(N, sample_cap, replace=False)
        sample = images[idx]
    else:
        sample = images
    items = list(sample)
    gz = parallel_map(gzip_one, items)
    pn = parallel_map(png_one, items)
    fg = parallel_map(filtered_gzip_one, items)
    wb = parallel_map(webp_one, items)
    wb_clean = [x for x in wb if x is not None]
    row = {
        "dataset": name, "n_images_used": N,
        "image_height": H, "image_width": W, "image_channels": C,
        "H_pixel": float(np.mean(Hpc)),
        "H_pixel_R": Hpc[0], "H_pixel_G": Hpc[1] if C>1 else "", "H_pixel_B": Hpc[2] if C>2 else "",
        "H_conditional": float(np.mean(Hcond)),
        "H_block_1x1": Hblk[1], "H_block_2x2": Hblk[2],
        "H_block_4x4": Hblk[4], "H_block_8x8": Hblk[8],
        "H_gzip_mean": float(np.mean(gz)), "H_gzip_std": float(np.std(gz)),
        "H_png_mean": float(np.mean(pn)),  "H_png_std": float(np.std(pn)),
        "H_filtered_gzip_mean": float(np.mean(fg)), "H_filtered_gzip_std": float(np.std(fg)),
        "H_jpeg_lossless_mean": float(np.mean(wb_clean)) if wb_clean else "",
        "elapsed_seconds": round(time.time()-t0,2),
        "notes": notes + (" webp_lossless_used_for_jpeg_lossless_field" if wb_clean else ""),
    }
    append_row(row)
    print(f"[{name}] done in {row['elapsed_seconds']}s  H_pixel={row['H_pixel']:.3f}  H_cond={row['H_conditional']:.3f}  H_png={row['H_png_mean']:.3f}", flush=True)
    return row

def load_cifar(which):
    from torchvision.datasets import CIFAR10, CIFAR100
    cls = CIFAR10 if which=="cifar10" else CIFAR100
    ds = cls(DATA, train=True, download=True)
    arr = ds.data  # (N,32,32,3) uint8
    return np.asarray(arr, dtype=np.uint8)

def load_stl10():
    from torchvision.datasets import STL10
    ds = STL10(DATA, split="train", download=True)
    # ds.data shape (N,3,96,96)
    arr = np.asarray(ds.data, dtype=np.uint8).transpose(0,2,3,1)
    return arr

def gen_random(n=10000):
    return np.random.RandomState(42).randint(0,256,size=(n,32,32,3),dtype=np.uint8)

def gen_const(n=1000):
    rs = np.random.RandomState(7)
    out = np.zeros((n,32,32,3),dtype=np.uint8)
    for i in range(n):
        out[i,:,:,:] = rs.randint(0,256,size=(3,),dtype=np.uint8)
    return out

def main():
    write_header()
    rows = []
    try:
        rows.append(measure("CIFAR-10", load_cifar("cifar10")))
    except Exception as e:
        print("CIFAR-10 failed:", e, flush=True)
    try:
        rows.append(measure("CIFAR-100", load_cifar("cifar100")))
    except Exception as e:
        print("CIFAR-100 failed:", e, flush=True)
    try:
        rows.append(measure("STL-10", load_stl10()))
    except Exception as e:
        print("STL-10 failed:", e, flush=True)
    rows.append(measure("RandomNoise", gen_random(), notes="control: max-entropy uniform noise"))
    rows.append(measure("ConstantColor", gen_const(), notes="control: min-entropy solid color"))
    print("ALL DONE", flush=True)

if __name__ == "__main__":
    main()
