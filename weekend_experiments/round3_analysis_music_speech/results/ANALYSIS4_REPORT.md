# Analysis 4: Music vs Speech

| Source | H_gzip (bits/sample) | Spectral entropy | Autocorrelation | Model throughput |
|---|---|---|---|---|
| speech | 15.605 | 9.50 | 0.5325 | 0.63 |
| music | 14.199 | 5.47 | 0.9904 | 2.83 |
| noise | 15.782 | 10.00 | 0.0008 | 0.00 |

## Why music > speech in throughput

Music has LOWER source entropy but HIGHER throughput → music's structure is more exploitable.
The mel-spectrogram representation may favor harmonic structure over formant structure.
