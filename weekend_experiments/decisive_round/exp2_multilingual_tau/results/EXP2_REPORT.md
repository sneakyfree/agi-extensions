# Exp 2: τ Across Languages and Domains

| Model | Corpus | BPT | Bits/char | Bits/byte | Bytes/char |
|---|---|---|---|---|---|
| Pythia-410M | english | 4.220 | 0.9526 | 0.9510 | 1.00 |
| Pythia-410M | python_code | 0.190 | 0.0691 | 0.0691 | 1.00 |
| Pythia-410M | dna_sequence | 4.433 | 2.0208 | 2.0208 | 1.00 |
| GPT-2-medium | english | 4.465 | 0.9975 | 0.9958 | 1.00 |
| GPT-2-medium | python_code | 0.785 | 0.3578 | 0.3578 | 1.00 |
| GPT-2-medium | dna_sequence | 4.094 | 2.1275 | 2.1275 | 1.00 |

## Key question: Is bits/char constant across languages?
If yes → the basin is about characters (cognitive units).
If bits/byte is constant → the basin is about byte-level compression.
