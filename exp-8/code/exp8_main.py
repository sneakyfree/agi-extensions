#!/usr/bin/env python3
"""Experiment 8: Vision throughput basin vs language."""
import os, sys, time, gc, io, gzip, math, json
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path("/home/user1-gpu/agi-extensions/exp-8")
RES = ROOT / "results"; PLT = ROOT / "plots"
RES.mkdir(exist_ok=True, parents=True); PLT.mkdir(exist_ok=True, parents=True)

device = "cuda"
LN2 = math.log(2)

VIT_MODELS = [
    "google/vit-base-patch16-224",
    "google/vit-large-patch16-224",
    "facebook/deit-base-patch16-224",
    "facebook/deit-small-patch16-224",
]

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def load_cifar100(n=None):
    from datasets import load_dataset
    ds = load_dataset("cifar100", split="test")
    if n: ds = ds.select(range(n))
    return ds

def get_processor_model(name):
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    proc = AutoImageProcessor.from_pretrained(name)
    model = AutoModelForImageClassification.from_pretrained(name, attn_implementation="eager").to(device).eval()
    return proc, model

# ImageNet-class label string to coarse mapping is hard. We'll measure cross-entropy
# on the model's own output distribution against the argmax (i.e. -log p(top1)),
# which is a surprise/confidence measure independent of ImageNet vs CIFAR labels.
# This still measures throughput: how many bits to specify the model's answer.

def exp_8a():
    log("=== EXP 8A: Vision throughput ===")
    ds = load_cifar100()
    rows = []
    for mname in VIT_MODELS:
        try:
            log(f"Loading {mname}")
            proc, model = get_processor_model(mname)
        except Exception as e:
            log(f"FAIL {mname}: {e}"); continue
        num_patches = 196
        bs = 32
        imgs_pil = [im.convert("RGB") for im in ds["img"]]
        N = len(imgs_pil)
        with torch.no_grad():
            for i in range(0, N, bs):
                batch = imgs_pil[i:i+bs]
                inputs = proc(images=batch, return_tensors="pt").to(device)
                out = model(**inputs, output_attentions=True)
                logits = out.logits
                logp = F.log_softmax(logits, dim=-1)
                p = logp.exp()
                # entropy of prediction distribution (bits)
                H_pred = -(p * logp).sum(-1) / LN2
                # surprise of top-1 prediction = -log p(top1) ; but to mirror "loss"
                # we use the predictive entropy as bits_per_image (information content
                # of the model's belief about this image's class).
                bits_per_image = H_pred  # already in bits
                bpp = bits_per_image / num_patches
                # attention entropy from last layer
                last_attn = out.attentions[-1]  # [B, heads, seq, seq]
                a = last_attn.clamp_min(1e-12)
                Hatt = -(a * a.log2()).sum(-1)  # over keys
                Hatt = Hatt.mean(dim=(1,2))  # avg heads & queries
                for j in range(len(batch)):
                    rows.append(dict(model=mname, image_id=i+j,
                                     bits_per_image=float(bits_per_image[j]),
                                     bits_per_patch=float(bpp[j]),
                                     attention_entropy=float(Hatt[j])))
                if i % (bs*20) == 0:
                    log(f"  {mname} {i}/{N}")
        del model, proc; gc.collect(); torch.cuda.empty_cache(); time.sleep(30)
    df = pd.DataFrame(rows)
    df.to_csv(RES/"exp8a_vision_throughput.csv", index=False)
    log(f"8A done: {len(df)} rows")
    return df

def shuffle_quadrants(arr):
    h,w,c = arr.shape; hh,ww = h//2,w//2
    qs = [arr[:hh,:ww], arr[:hh,ww:], arr[hh:,:ww], arr[hh:,ww:]]
    idx = np.random.permutation(4)
    qs = [qs[i] for i in idx]
    top = np.concatenate([qs[0], qs[1]], axis=1)
    bot = np.concatenate([qs[2], qs[3]], axis=1)
    return np.concatenate([top, bot], axis=0)

def shuffle_patches(arr, ps=16):
    h,w,c = arr.shape
    nh,nw = h//ps, w//ps
    patches = arr.reshape(nh, ps, nw, ps, c).transpose(0,2,1,3,4).reshape(nh*nw, ps, ps, c)
    idx = np.random.permutation(nh*nw)
    patches = patches[idx].reshape(nh, nw, ps, ps, c).transpose(0,2,1,3,4).reshape(h,w,c)
    return patches

def shuffle_rows(arr):
    idx = np.random.permutation(arr.shape[0])
    return arr[idx]

def shuffle_pixels(arr):
    h,w,c = arr.shape
    flat = arr.reshape(-1, c)
    idx = np.random.permutation(flat.shape[0])
    return flat[idx].reshape(h,w,c)

def exp_8b():
    log("=== EXP 8B: Image shuffling ===")
    ds = load_cifar100(n=500)
    imgs_pil = [im.convert("RGB").resize((224,224)) for im in ds["img"]]
    arrs = [np.array(im) for im in imgs_pil]
    conditions = ["original", "quadrants", "patches16", "rows", "pixels"]
    rows = []
    np.random.seed(0)
    versions = {c: [] for c in conditions}
    for a in arrs:
        versions["original"].append(a)
        versions["quadrants"].append(shuffle_quadrants(a))
        versions["patches16"].append(shuffle_patches(a))
        versions["rows"].append(shuffle_rows(a))
        versions["pixels"].append(shuffle_pixels(a))
    for mname in VIT_MODELS:
        try:
            log(f"Loading {mname}")
            proc, model = get_processor_model(mname)
        except Exception as e:
            log(f"FAIL {mname}: {e}"); continue
        bs = 32
        with torch.no_grad():
            for cond in conditions:
                pil_list = [Image.fromarray(a) for a in versions[cond]]
                N = len(pil_list)
                for i in range(0, N, bs):
                    batch = pil_list[i:i+bs]
                    inputs = proc(images=batch, return_tensors="pt").to(device)
                    out = model(**inputs)
                    logp = F.log_softmax(out.logits, -1)
                    p = logp.exp()
                    H = -(p*logp).sum(-1)/LN2
                    for j in range(len(batch)):
                        rows.append(dict(model=mname, condition=cond,
                                         image_id=i+j, bits_per_image=float(H[j])))
                log(f"  {mname} {cond} done")
        del model, proc; gc.collect(); torch.cuda.empty_cache(); time.sleep(30)
    df = pd.DataFrame(rows)
    df.to_csv(RES/"exp8b_image_shuffling.csv", index=False)
    log("8B done")
    return df

def exp_8c(df_8a):
    log("=== EXP 8C: Cross-modal density ===")
    rows = []
    # Text
    try:
        from datasets import load_dataset
        wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n".join(wt["text"])[:200000]
        raw = text.encode("utf-8")
        gz = gzip.compress(raw)
        text_bpb = 8 * len(gz) / len(raw)  # bits per byte
        rows.append(dict(modality="text", metric="raw_gzip_bpb", value=text_bpb))
        # pythia
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            tk = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")
            mdl = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-410m").to(device).eval()
            ids = tk(text[:50000], return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                # chunked
                losses = []
                CL = 1024
                for i in range(0, ids.shape[1]-1, CL):
                    chunk = ids[:, i:i+CL+1]
                    if chunk.shape[1] < 2: break
                    out = mdl(chunk[:, :-1], labels=chunk[:, 1:])
                    losses.append(float(out.loss))
                bpt = np.mean(losses)/LN2
                rows.append(dict(modality="text", metric="pythia410m_bpt", value=bpt))
            del mdl, tk; gc.collect(); torch.cuda.empty_cache()
        except Exception as e:
            log(f"pythia fail: {e}")
    except Exception as e:
        log(f"text fail: {e}")

    # Images: gzip BMP
    ds = load_cifar100(n=500)
    ratios = []; bpp_list = []
    for im in ds["img"]:
        im2 = im.convert("RGB").resize((224,224))
        buf = io.BytesIO(); im2.save(buf, format="BMP")
        raw = buf.getvalue()
        gz = gzip.compress(raw)
        ratio = len(gz)/len(raw)
        ratios.append(ratio)
        bpp_list.append(24*ratio)
    rows.append(dict(modality="vision", metric="gzip_compress_ratio_mean", value=float(np.mean(ratios))))
    rows.append(dict(modality="vision", metric="raw_bits_per_pixel_mean", value=float(np.mean(bpp_list))))
    # bits per patch from 8A
    if df_8a is not None and len(df_8a):
        for m, sub in df_8a.groupby("model"):
            rows.append(dict(modality="vision", metric=f"model_bits_per_patch::{m}", value=float(sub.bits_per_patch.mean())))
            rows.append(dict(modality="vision", metric=f"model_bits_per_image::{m}", value=float(sub.bits_per_image.mean())))
    df = pd.DataFrame(rows)
    df.to_csv(RES/"exp8c_information_density.csv", index=False)
    log("8C done")
    return df

def exp_8d():
    log("=== EXP 8D: Multimodal (best effort) ===")
    rows = []
    try:
        from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        mid = "llava-hf/llava-1.5-7b-hf"
        proc = AutoProcessor.from_pretrained(mid)
        model = LlavaForConditionalGeneration.from_pretrained(mid, quantization_config=bnb, device_map="auto").eval()
        # text-only BPT
        text = "The capital of France is Paris. Machine learning models trained on large corpora develop emergent capabilities."
        inputs = proc(text=text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs, labels=inputs.input_ids)
            text_bpt = float(out.loss)/LN2
        rows.append(dict(condition="text_only", bpt=text_bpt))
        # image+text
        ds = load_cifar100(n=10)
        bpts = []
        for im in ds["img"]:
            im2 = im.convert("RGB").resize((224,224))
            prompt = "USER: <image>\nDescribe this image.\nASSISTANT: An image."
            inputs = proc(images=im2, text=prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**inputs, labels=inputs.input_ids)
                bpts.append(float(out.loss)/LN2)
        rows.append(dict(condition="image_text", bpt=float(np.mean(bpts))))
        del model, proc; gc.collect(); torch.cuda.empty_cache()
    except Exception as e:
        log(f"8D skipped: {e}")
        rows.append(dict(condition="skipped", bpt=float('nan')))
    df = pd.DataFrame(rows)
    df.to_csv(RES/"exp8d_multimodal.csv", index=False)
    return df

def make_plots(df_a, df_b, df_c):
    sns.set_style("whitegrid")
    # Plot 1: bits-per-patch histogram
    fig, ax = plt.subplots(figsize=(9,5))
    for m, sub in df_a.groupby("model"):
        ax.hist(sub.bits_per_patch, bins=50, alpha=0.5, label=m.split("/")[-1])
    ax.axvspan(3, 6, alpha=0.2, color="red", label="Language basin [3,6] BPT")
    ax.set_xlabel("bits per patch"); ax.set_ylabel("count"); ax.legend(fontsize=8)
    ax.set_title("Vision: bits-per-patch distribution vs language basin")
    plt.tight_layout(); plt.savefig(PLT/"plot1_bpp_histogram.png", dpi=120); plt.close()

    # Plot 2: shuffle cascade
    fig, ax = plt.subplots(figsize=(9,5))
    order = ["original","quadrants","patches16","rows","pixels"]
    for m, sub in df_b.groupby("model"):
        means = [sub[sub.condition==c].bits_per_image.mean() for c in order]
        ax.plot(order, means, marker="o", label=m.split("/")[-1])
    ax.set_xlabel("shuffle level"); ax.set_ylabel("bits per image")
    ax.set_title("Image shuffling cascade"); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(PLT/"plot2_shuffle_cascade.png", dpi=120); plt.close()

    # Plot 3: structural bonus
    fig, ax = plt.subplots(figsize=(9,5))
    bonuses = {}
    for m, sub in df_b.groupby("model"):
        orig = sub[sub.condition=="original"].bits_per_image.mean()
        bonuses[m.split("/")[-1]] = sub[sub.condition=="pixels"].bits_per_image.mean() - orig
    names = list(bonuses.keys()) + ["language_total","language_syntax"]
    vals  = list(bonuses.values()) + [6.7, 3.3]
    colors = ["steelblue"]*len(bonuses) + ["darkred","orange"]
    ax.bar(names, vals, color=colors)
    ax.set_ylabel("structural bonus (bits)")
    ax.set_title("Structural bonus: vision vs language")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout(); plt.savefig(PLT/"plot3_structural_bonus.png", dpi=120); plt.close()

    # Plot 4: cross-modal density scatter
    fig, ax = plt.subplots(figsize=(7,7))
    text_raw = df_c[df_c.metric=="raw_gzip_bpb"].value.values
    text_mod = df_c[df_c.metric=="pythia410m_bpt"].value.values
    vis_raw = df_c[df_c.metric=="raw_bits_per_pixel_mean"].value.values
    vis_mods = df_c[df_c.metric.str.startswith("model_bits_per_patch::")].value.values
    if len(text_raw) and len(text_mod):
        ax.scatter(text_raw, text_mod, s=120, c="darkred", label="text (bpb / BPT)")
    if len(vis_raw) and len(vis_mods):
        ax.scatter([vis_raw[0]]*len(vis_mods), vis_mods, s=120, c="steelblue", label="vision (bpp / b/patch)")
    lim = max(10, float(vis_raw[0]) if len(vis_raw) else 10)
    ax.plot([0,lim],[0,lim],"k--",alpha=0.4,label="1:1")
    ax.set_xlabel("raw entropy"); ax.set_ylabel("model throughput")
    ax.legend(); ax.set_title("Cross-modal: model throughput vs raw entropy")
    plt.tight_layout(); plt.savefig(PLT/"plot4_cross_modal.png", dpi=120); plt.close()

def main():
    t0 = time.time()
    df_a = exp_8a()
    df_b = exp_8b()
    df_c = exp_8c(df_a)
    df_d = exp_8d()
    make_plots(df_a, df_b, df_c)
    # Summary
    summary = {}
    summary["bits_per_patch_by_model"] = df_a.groupby("model").bits_per_patch.mean().to_dict()
    summary["bits_per_image_by_model"] = df_a.groupby("model").bits_per_image.mean().to_dict()
    summary["attention_entropy_by_model"] = df_a.groupby("model").attention_entropy.mean().to_dict()
    summary["shuffle_means"] = df_b.groupby(["model","condition"]).bits_per_image.mean().reset_index().to_dict("records")
    summary["density_rows"] = df_c.to_dict("records")
    summary["multimodal"] = df_d.to_dict("records")
    summary["elapsed_sec"] = time.time()-t0
    with open(RES/"summary.json","w") as f: json.dump(summary, f, indent=2, default=str)
    log(f"ALL DONE in {summary['elapsed_sec']:.0f}s")
    print(json.dumps(summary, indent=2, default=str))

if __name__ == "__main__":
    main()
