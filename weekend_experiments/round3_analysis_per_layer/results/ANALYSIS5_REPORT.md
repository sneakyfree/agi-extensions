# Analysis 5: Per-Layer Cliff Progression

Tested attention.dense weight across all 24 layers of Pythia-410M.

| Layers | Mean cliff ratio |
|---|---|
| 0–7 (early) | 4.65× |
| 8–15 (middle) | 4.68× |
| 16–23 (late) | 3.78× |

**The cliff is roughly constant across layers.** Depth doesn't matter much.
