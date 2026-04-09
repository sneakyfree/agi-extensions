# Paper 7 — Publications Card

**Badge:** 7
**Title:** The Throughput Basin Origin
**Subtitle:** Different data, different basin

**Authors:** Windstorm Institute
**Status:** Preprint, 2026

**Abstract (card-length):**
Prior work in this arc identified a stable ~4 bits-per-token throughput basin across families and scales. Paper 7 asks where the basin comes from. We train SYN-8 — an architecturally identical transformer — on a synthetic corpus engineered to carry ~8 bits of entropy per token. SYN-8 does not converge to ~4 BPT. It tracks the 8-bit ceiling of its data. The throughput basin is therefore not a constraint of the architecture, the optimizer, or the scale; it is the compressibility floor of natural text reflected back through the model. The wall was a mirror.

**Key result:** SYN-8 tracks 8-bit entropy of its training distribution rather than compressing to the ~4 BPT basin observed for natural-language–trained models of the same architecture.
