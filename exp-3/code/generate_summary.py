#!/usr/bin/env python3
"""
Generate SUMMARY.md for Experiment 3
"""

import pandas as pd
from pathlib import Path
import numpy as np

RESULTS_DIR = Path('/home/user1-gpu/agi-extensions/exp-3/results')
EXP_DIR = Path('/home/user1-gpu/agi-extensions/exp-3')

def generate_summary():
    """Generate comprehensive summary document"""

    # Load results
    df_bpt = pd.read_csv(RESULTS_DIR / 'exp3_bpt_comparison.csv')
    df_shuffling = pd.read_csv(RESULTS_DIR / 'exp3_shuffling_cascade.csv')
    df_seven = pd.read_csv(RESULTS_DIR / 'exp3_seven_corpus.csv')
    df_energy = pd.read_csv(RESULTS_DIR / 'exp3_energy.csv')
    df_stats = pd.read_csv(RESULTS_DIR / 'exp3_statistics.csv')

    summary = []
    summary.append("# Experiment 3: Recurrent vs Transformer Architecture Comparison")
    summary.append("")
    summary.append("**Windstorm Institute Paper 7 - AGI Extensions of the Throughput Basin Framework**")
    summary.append("")
    summary.append("## Research Question")
    summary.append("")
    summary.append("Do truly serial architectures (state-space models like Mamba/RWKV) show tighter convergence")
    summary.append("to τ = 4.16 bits with less variance across corpora than transformers?")
    summary.append("")
    summary.append("## Hypothesis")
    summary.append("")
    summary.append("**H1:** Mamba/RWKV models show BPT closer to 4.16 with less variance across corpora than")
    summary.append("parameter-matched transformers, because they cannot parallelize around the serial decoding bottleneck.")
    summary.append("")
    summary.append("**H0:** No significant difference between architectures.")
    summary.append("")
    summary.append("## Methodology")
    summary.append("")
    summary.append("### Models Tested")
    summary.append("")
    summary.append("**Transformers:**")
    for model in df_bpt[df_bpt['arch_type'] == 'transformer']['model'].values:
        params = df_bpt[df_bpt['model'] == model]['params'].values[0]
        summary.append(f"- {model} ({params:,} parameters)")
    summary.append("")
    summary.append("**Serial/SSM:**")
    for model in df_bpt[df_bpt['arch_type'] == 'serial']['model'].values:
        params = df_bpt[df_bpt['model'] == model]['params'].values[0]
        summary.append(f"- {model} ({params:,} parameters)")
    summary.append("")
    summary.append("### Measurements")
    summary.append("")
    summary.append("1. **BPT and BPB** on WikiText-2 test set (3 runs, mean ± std)")
    summary.append("2. **Shuffling cascade** - 5 levels from original to fully shuffled")
    summary.append("3. **Seven-corpus battery** - WikiText-2, Python, DNA, Shuffled, Math, Random ASCII, CSV")
    summary.append("4. **Energy per token** - nvidia-smi power monitoring")
    summary.append("")
    summary.append("## Key Findings")
    summary.append("")

    # BPT comparison
    summary.append("### 1. BPT on WikiText-2")
    summary.append("")
    summary.append("| Model | Architecture | Parameters | BPT (mean ± std) | BPB (mean ± std) |")
    summary.append("|-------|--------------|------------|------------------|------------------|")

    for _, row in df_bpt.iterrows():
        summary.append(f"| {row['model'].split('/')[-1]} | {row['arch_type']} | {row['params']:,} | "
                      f"{row['bpt_mean']:.3f} ± {row['bpt_std']:.3f} | "
                      f"{row['bpb_mean']:.3f} ± {row['bpb_std']:.3f} |")

    summary.append("")

    # Architecture comparison
    transformer_mean = df_bpt[df_bpt['arch_type'] == 'transformer']['bpt_mean'].mean()
    serial_mean = df_bpt[df_bpt['arch_type'] == 'serial']['bpt_mean'].mean()

    summary.append(f"**Transformer mean BPT:** {transformer_mean:.3f} bits")
    summary.append(f"**Serial/SSM mean BPT:** {serial_mean:.3f} bits")
    summary.append(f"**Difference:** {abs(transformer_mean - serial_mean):.3f} bits")
    summary.append("")

    # Check basin convergence
    basin_min, basin_max = 3.0, 6.0
    tau = 4.16

    transformer_in_basin = df_bpt[df_bpt['arch_type'] == 'transformer']['bpt_mean'].apply(
        lambda x: basin_min <= x <= basin_max).sum()
    serial_in_basin = df_bpt[df_bpt['arch_type'] == 'serial']['bpt_mean'].apply(
        lambda x: basin_min <= x <= basin_max).sum()

    transformer_total = len(df_bpt[df_bpt['arch_type'] == 'transformer'])
    serial_total = len(df_bpt[df_bpt['arch_type'] == 'serial'])

    summary.append(f"**Transformer models in basin [3,6] bits:** {transformer_in_basin}/{transformer_total} "
                  f"({100*transformer_in_basin/transformer_total:.1f}%)")
    summary.append(f"**Serial models in basin [3,6] bits:** {serial_in_basin}/{serial_total} "
                  f"({100*serial_in_basin/serial_total:.1f}%)")
    summary.append("")

    # Variance across corpora
    summary.append("### 2. Variance Across 7 Corpora")
    summary.append("")

    transformer_corpus_bpt = df_seven[df_seven['arch_type'] == 'transformer']['bpt'].values
    serial_corpus_bpt = df_seven[df_seven['arch_type'] == 'serial']['bpt'].values

    transformer_var = np.var(transformer_corpus_bpt)
    serial_var = np.var(serial_corpus_bpt)

    summary.append(f"**Transformer BPT variance:** {transformer_var:.3f}")
    summary.append(f"**Serial/SSM BPT variance:** {serial_var:.3f}")
    summary.append(f"**Ratio:** {transformer_var/serial_var:.3f}x" if serial_var > 0 else "N/A")
    summary.append("")

    if serial_var < transformer_var:
        summary.append("✓ Serial architectures show **lower variance** (tighter basin constraint)")
    else:
        summary.append("✗ Transformers show lower or equal variance")
    summary.append("")

    # Structural bonus
    summary.append("### 3. Structural Bonus (All Shuffled - Original)")
    summary.append("")

    transformer_bonus = []
    serial_bonus = []

    for model in df_shuffling['model'].unique():
        model_data = df_shuffling[df_shuffling['model'] == model]
        original = model_data[model_data['shuffle_level'] == 'original']['bpt'].values
        shuffled = model_data[model_data['shuffle_level'] == 'all_shuffled']['bpt'].values

        if len(original) > 0 and len(shuffled) > 0:
            bonus = shuffled[0] - original[0]
            arch_type = model_data['arch_type'].iloc[0]

            if arch_type == 'transformer':
                transformer_bonus.append(bonus)
            else:
                serial_bonus.append(bonus)

    summary.append(f"**Transformer structural bonus:** {np.mean(transformer_bonus):.3f} ± {np.std(transformer_bonus):.3f} bits")
    summary.append(f"**Serial/SSM structural bonus:** {np.mean(serial_bonus):.3f} ± {np.std(serial_bonus):.3f} bits")
    summary.append("")

    # Energy efficiency
    summary.append("### 4. Energy Efficiency")
    summary.append("")

    df_merged = df_bpt.merge(df_energy, on=['model', 'arch_type'])

    summary.append("| Model | Architecture | Energy/Token (mJ) | Bits per Joule |")
    summary.append("|-------|--------------|-------------------|----------------|")

    for _, row in df_merged.iterrows():
        bits_per_joule = row['bpt_mean'] / (row['energy_per_token_mJ'] / 1000) if row['energy_per_token_mJ'] > 0 else 0
        summary.append(f"| {row['model'].split('/')[-1]} | {row['arch_type']} | "
                      f"{row['energy_per_token_mJ']:.3f} | {bits_per_joule:.1f} |")

    summary.append("")

    # Statistical tests
    summary.append("## Statistical Analysis")
    summary.append("")

    for _, row in df_stats.iterrows():
        summary.append(f"### {row['test']}")
        summary.append("")
        summary.append(f"- **Statistic:** {row['statistic']:.4f}")
        if not pd.isna(row['p_value']):
            summary.append(f"- **p-value:** {row['p_value']:.4f}")
        summary.append(f"- **Significant:** {'Yes' if row['significant'] else 'No'} (α = 0.05)")
        summary.append(f"- **Interpretation:** {row['interpretation']}")
        summary.append("")

    # Conclusion
    summary.append("## Conclusion")
    summary.append("")

    # Determine which hypothesis is supported
    welch_test = df_stats[df_stats['test'].str.contains("Welch")]['significant'].values
    levene_test = df_stats[df_stats['test'].str.contains("Levene")]['significant'].values

    h1_evidence = []
    h0_evidence = []

    # Check mean difference
    if len(welch_test) > 0 and welch_test[0]:
        if abs(serial_mean - tau) < abs(transformer_mean - tau):
            h1_evidence.append("Serial models closer to τ = 4.16")
        else:
            h0_evidence.append("No clear convergence advantage for serial models")
    else:
        h0_evidence.append("No significant difference in mean BPT")

    # Check variance
    if len(levene_test) > 0 and levene_test[0]:
        if serial_var < transformer_var:
            h1_evidence.append("Serial models show tighter variance (stronger basin constraint)")
        else:
            h0_evidence.append("Transformers show equal or tighter variance")
    else:
        h0_evidence.append("No significant difference in BPT variance")

    summary.append("**Evidence for H1 (Serial tighter):**")
    if h1_evidence:
        for evidence in h1_evidence:
            summary.append(f"- {evidence}")
    else:
        summary.append("- None")
    summary.append("")

    summary.append("**Evidence for H0 (No difference):**")
    if h0_evidence:
        for evidence in h0_evidence:
            summary.append(f"- {evidence}")
    else:
        summary.append("- None")
    summary.append("")

    if len(h1_evidence) > len(h0_evidence):
        summary.append("**VERDICT: H1 SUPPORTED** - Serial architectures show tighter convergence to the throughput basin.")
        summary.append("")
        summary.append("This suggests the basin constraint may be partially **architectural** - intrinsic to serial")
        summary.append("processing mechanisms rather than purely inherited from training data.")
    elif len(h0_evidence) > len(h1_evidence):
        summary.append("**VERDICT: H0 SUPPORTED** - No significant difference between architectures.")
        summary.append("")
        summary.append("This suggests the basin constraint is **data-driven** rather than architectural - both")
        summary.append("transformers and serial models inherit ~4.4 bits/token from biological training corpora.")
    else:
        summary.append("**VERDICT: MIXED EVIDENCE** - Results show some support for both hypotheses.")
        summary.append("")
        summary.append("The throughput basin may emerge from a **combination** of data constraints and")
        summary.append("architectural properties of serial decoding.")

    summary.append("")
    summary.append("## Implications for Paper 7")
    summary.append("")
    summary.append("1. **Architecture vs Data:** The results help isolate whether the basin is architectural or data-driven")
    summary.append("2. **Model Selection:** Identifies which architectures may be more thermodynamically efficient for AGI")
    summary.append("3. **Future Work:** Experiment 1 (synthetic training) will provide definitive test by training on non-biological data")
    summary.append("")

    # Write to file
    output_file = EXP_DIR / 'SUMMARY.md'
    with open(output_file, 'w') as f:
        f.write('\n'.join(summary))

    print(f"Summary written to {output_file}")

if __name__ == '__main__':
    generate_summary()
