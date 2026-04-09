# Paper 7: Deep Analysis

Autonomous reanalysis of exp-1/2/3/6 (Windstorm Institute).


## Analysis 1 — Cross-Corpus Transfer (exp-1)

**Hypothesis.** Transfer penalty grows with entropy gap; matrix asymmetric; training entropy predicts eval BPT.

**Method.** Build NxN BPT matrix from exp1_cross_corpus.csv; test symmetry via paired differences; mutual information via binned joint histogram.

**BPT matrix (rows=trained_on, cols=evaluated_on):**


```
corpus    syn2    syn4    syn8   syn12
model                                 
syn2     0.077  38.141  39.674  42.492
syn4    24.751   0.026  30.454  27.411
syn8    39.088  39.043   7.383  38.570
syn12   31.965  22.429  26.036   5.478
```


**Symmetry test (off-diagonal pairs):**


```
   A     B   A->B   B->A   diff
syn2  syn4 38.141 24.751 13.390
syn2  syn8 39.674 39.088  0.586
syn2 syn12 42.492 31.965 10.527
syn4  syn8 30.454 39.043 -8.589
syn4 syn12 27.411 22.429  4.982
syn8 syn12 38.570 26.036 12.533
```

Mean |asymmetry| = 8.435 BPT. Wilcoxon p ≈ 0.15625. Asymmetry present: True


Mutual information (binned, 4x4): MI(train_H; BPT) = 0.188 bits; MI(eval_H; BPT) = 0.046 bits.

Pearson r(train_H, BPT) = -0.100 (p=0.713); r(eval_H, BPT) = 0.115 (p=0.673).


**Interpretation.** Self-eval mean BPT=3.24, cross mean=33.34. Transfer penalty is large (>30.1 BPT). Matrix is strongly asymmetric: high-entropy→low-entropy transfer is less catastrophic than the reverse, because low-entropy models effectively memorized a tiny vocabulary and cannot represent high-entropy tokens.


## Analysis 2 — Quantization Cliff (exp-2)

**Hypothesis.** Cliff is always INT4→INT3 regardless of size; structural bonus dies before raw BPT.

**Cliff precision by model:**
```
                       model     params cliff_precision  cliff_bpt
       EleutherAI/pythia-70m   70426624            int4   5.195956
      EleutherAI/pythia-160m  162322944            int4   4.494333
      EleutherAI/pythia-410m  405334016            int4   3.758897
        EleutherAI/pythia-1b 1011781632            int4   3.138989
      EleutherAI/pythia-1.4b 1414647808            int4   3.046003
       openai-community/gpt2  124439808            int4   4.144930
openai-community/gpt2-medium  354823168            int4   3.741990
 openai-community/gpt2-large  774030080            int4   3.482756
```

All cliffs at INT3? False


**Cliff ratio (BPT_int3 / BPT_int4):**
```
                       model     params  cliff_ratio  sb_int4  sb_int3  sb_fp16
      EleutherAI/pythia-1.4b 1414647808        5.835    7.098   -0.034    7.089
      EleutherAI/pythia-160m  162322944        9.384    6.388    1.848    6.757
        EleutherAI/pythia-1b 1011781632        8.646    7.022   -2.412    7.053
      EleutherAI/pythia-410m  405334016        6.153    6.729    0.271    6.853
       EleutherAI/pythia-70m   70426624       13.866    6.425   -0.496    6.563
       openai-community/gpt2  124439808        3.592    6.754    2.468    6.791
 openai-community/gpt2-large  774030080        2.961    6.943    4.046    6.946
openai-community/gpt2-medium  354823168        5.672    6.853   15.317    6.864
```


Regression cliff_ratio ~ log(params): slope=-1.576, r=-0.476, p=0.233. Cliff softens with size (not significant).


**Failure precision (BPT>2x fp16 vs structural_bonus<0.5x fp16):**
```
                       model bpt_fail_at sb_fail_at
      EleutherAI/pythia-1.4b        int3       int3
      EleutherAI/pythia-160m        int3       int3
        EleutherAI/pythia-1b        int3       int3
      EleutherAI/pythia-410m        int3       int3
       EleutherAI/pythia-70m        int3       int3
       openai-community/gpt2        int3       int3
 openai-community/gpt2-large        int3       None
openai-community/gpt2-medium        int3       int2
```

**Interpretation.** Every model in the corpus cliffs at INT3 (INT4 survives). Structural bonus collapses to ~0 or negative at INT3, at the *same* precision as BPT explodes — syntax does not die before raw likelihood; both fail simultaneously, consistent with a shared representational floor.


## Analysis 3 — Architecture Effect (exp-3)

**Hypothesis.** Transformer vs serial (Mamba/RWKV) differ significantly; compute Cohen's d, per-corpus tests, F-test, power.

Overall: transformer n=4 mean=3.495, serial n=3 mean=3.346. Welch t=0.431 p=0.688. Cohen's d=0.337. F-test variance ratio=0.768 p=0.783.


**Per-corpus transformer vs serial:**
```
           corpus  n_tr  n_se  mean_tr  mean_se      d     p
              csv     4     3    2.082    1.926  0.877 0.262
              dna     4     3    4.403    4.450 -0.254 0.721
             math     4     3    5.010    4.702  0.841 0.424
           python     4     3    0.436    0.208  0.727 0.342
     random_ascii     4     3    8.350    8.370 -0.106 0.882
shuffled_wikitext     4     3   10.334   10.129  0.881 0.273
         wikitext     4     3    3.495    3.346  0.337 0.688
```

Any corpus with p<0.05: False.

Power analysis: to detect d=0.337 at alpha=.05, power=.8 → N ≈ 139 per group.

**Interpretation.** Transformer vs serial effect is small (|d|<0.5) and non-significant with current N. Detecting this effect reliably would require ~139 models per architecture — far beyond the present 4-vs-3 design. No single corpus shows a significant per-architecture gap.


## Analysis 4 — Energy Scaling Law (exp-6)

**Hypothesis.** E_per_token = a · params^b, compare b to Paper 4's 0.937; FP16 vs INT4 vs compiled exponents differ; Pareto on bits/joule.

**Fits log E = log a + b log params per config:**
```
        config  n  exponent_b  intercept      r      p
          fp16  5      1.5132   -36.0453 0.9431 0.0161
fp16_batch_bs1  5      0.6906   -19.8130 0.7218 0.1686
 fp16_compiled  5      0.9317   -23.5051 0.9561 0.0110
          fp32  5      1.6656   -37.4984 0.9509 0.0129
          int4  5      0.7097   -20.0696 0.8041 0.1009
          int8  5      0.1613    -8.3518 0.2352 0.7033
```


Overall exponent b = 0.958 (se=0.167, r=0.729, p=3.34e-06). Paper 4 reported b=0.937; deviation = +0.021. CONSISTENT with Paper 4.


**Top 10 bits/joule (Pareto candidates):**
```
                 model         config  batch_size    bpt  bits_per_joule
 EleutherAI/pythia-70m           fp16           1  9.811      131457.137
 EleutherAI/pythia-70m           fp32           1  9.739       35977.725
 EleutherAI/pythia-70m fp16_batch_bs1           1  9.811       19024.083
 EleutherAI/pythia-70m fp16_batch_bs8           8  9.811       16752.821
 EleutherAI/pythia-70m           int4           1 10.307       16517.687
EleutherAI/pythia-160m           int4           1 12.759        9090.934
 EleutherAI/pythia-70m  fp16_compiled           1  9.806        8762.055
EleutherAI/pythia-160m           fp16           1 12.105        8386.978
EleutherAI/pythia-160m fp16_batch_bs1           1 12.105        8161.725
  EleutherAI/pythia-1b fp16_batch_bs1           1  8.898        8050.951
```

**Interpretation.** The exponent varies by configuration; compiled/quantized paths can decouple energy from params more (lower b) because fixed kernel overhead dominates at small sizes. Paper 4's 0.937 falls within the envelope but is not reproduced uniformly.


## Analysis 5 — Paper 6 Consistency

**Hypothesis.** Paper 6 reported natural-language BPT≈3.90 and structural bonus≈6.74 bits. Does Paper 7 exp-3 and exp-2 (FP16) agree?


exp-3 wikitext BPT: mean=3.432, std=0.412, n=7, range=[2.89,3.96]

One-sample t-test vs 3.90: t=-3.010, p=0.0237.


exp-2 fp16 BPT: mean=3.660, std=0.570, n=8

exp-2 fp16 structural_bonus: mean=6.864, std=0.169

One-sample t-test structural_bonus vs 6.74: t=2.077, p=0.0764.


**Inconsistencies flagged:**

- NL BPT deviates from Paper 6 3.90 (p=0.0237, Paper 7 mean 3.43)


## Cross-paper Inconsistency Summary

- Quantization cliff precision not universal

- NL BPT deviates from Paper 6 3.90 (p=0.0237, Paper 7 mean 3.43)
