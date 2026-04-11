# R5 Scale Test Report

## Verdict

THESIS FALSIFIED AT SCALE. 1B SYN-8 BPT = 3.824 (< 5.0). The architectural hypothesis re-enters.

## Results

| Model | Params | Seed | BPT | BPSS* | BPSS*/H | Time (hrs) | Peak VRAM |
|---|---|---|---|---|---|---|---|
| gpt2_92m | 85,790,208 | 42 | 3.8224 | 2.0002 | 0.2500 | 1.88 | 1764 MB |
| gpt2_92m | 85,790,208 | 137 | 3.8224 | 2.0002 | 0.2500 | 1.89 | 1764 MB |
| gpt2_1b | 1,210,560,512 | 42 | 3.8242 | 2.0012 | 0.2501 | 5.23 | 23118 MB |
| gpt2_1b | 1,210,560,512 | 137 | 3.8229 | 2.0005 | 0.2501 | 5.25 | 23118 MB |

## Interpretation

THESIS FALSIFIED AT SCALE. 1B SYN-8 BPT = 3.824 (< 5.0). The architectural hypothesis re-enters.

Source entropy H = 8.0 bits/symbol. If BPT/H ≈ 1.0, the model perfectly tracks source entropy.
