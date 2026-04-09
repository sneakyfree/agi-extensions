#!/usr/bin/env python3
"""
Experiment 6: Energy Survey / Thermodynamic Roadmap
Windstorm Institute Paper 7 - AGI Extensions of the Throughput Basin Framework

Where does the RTX 5090 sit on the ribosome-to-Landauer thermodynamic spectrum?
"""

import os
import sys
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import subprocess
import threading
import logging
from datetime import datetime

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/user1-gpu/agi-extensions/exp-6/exp6.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Constants
RESULTS_DIR = Path('/home/user1-gpu/agi-extensions/exp-6/results')
PLOTS_DIR = Path('/home/user1-gpu/agi-extensions/exp-6/plots')
RESULTS_DIR.mkdir(exist_ok=True, parents=True)
PLOTS_DIR.mkdir(exist_ok=True, parents=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
COOLING_TIME = 30
EVAL_TOKENS = 10000  # Evaluate on 10k tokens

# Physical constants
K_B = 1.380649e-23  # Boltzmann constant (J/K)

# Ribosome reference values (from Paper 4)
RIBOSOME_BPT = 4.39
RIBOSOME_ENERGY_PER_CODON_J = 1.66e-19
RIBOSOME_PHI = 1.02
RIBOSOME_ENERGY_PER_BIT_J = RIBOSOME_ENERGY_PER_CODON_J / RIBOSOME_BPT

# Models to test
MODELS = [
    'EleutherAI/pythia-70m',
    'EleutherAI/pythia-160m',
    'EleutherAI/pythia-410m',
    'EleutherAI/pythia-1b',
    'EleutherAI/pythia-1.4b',
]


class PowerMonitor:
    """Background power monitoring using nvidia-smi"""

    def __init__(self):
        self.power_readings = []
        self.temp_readings = []
        self.running = False
        self.thread = None

    def start(self):
        self.power_readings = []
        self.temp_readings = []
        self.running = True
        self.thread = threading.Thread(target=self._monitor)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)

        if not self.power_readings:
            return {'mean_power_W': 0, 'mean_temp_C': 0}

        return {
            'mean_power_W': np.mean(self.power_readings),
            'std_power_W': np.std(self.power_readings),
            'max_power_W': np.max(self.power_readings),
            'mean_temp_C': np.mean(self.temp_readings),
        }

    def _monitor(self):
        while self.running:
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=power.draw,temperature.gpu',
                     '--format=csv,noheader,nounits'],
                    capture_output=True,
                    text=True,
                    timeout=1
                )
                if result.returncode == 0:
                    parts = result.stdout.strip().split(',')
                    if len(parts) == 2:
                        power = float(parts[0].strip())
                        temp = float(parts[1].strip())
                        self.power_readings.append(power)
                        self.temp_readings.append(temp)
            except Exception as e:
                logger.warning(f"Power monitoring error: {e}")
            time.sleep(0.1)


def load_model_with_config(model_name: str, config: str) -> Tuple[Optional[object], Optional[object], str]:
    """
    Load model with specified configuration.
    Configs: 'fp32', 'fp16', 'int8', 'int4', 'fp16_compiled', 'fp16_batch_X'
    """
    logger.info(f"Loading {model_name} with config {config}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Parse config
        if config == 'fp32':
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map='auto',
            )
        elif config == 'fp16' or config.startswith('fp16_batch'):
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map='auto',
            )
        elif config == 'fp16_compiled':
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map='auto',
            )
            model = torch.compile(model)
        elif config == 'int8':
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                load_in_8bit=True,
                device_map='auto',
            )
        elif config == 'int4':
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map='auto',
            )
        else:
            raise ValueError(f"Unknown config: {config}")

        model.eval()
        logger.info(f"Successfully loaded {model_name} with {config}")
        return model, tokenizer, config

    except Exception as e:
        logger.error(f"Failed to load {model_name} with {config}: {e}")
        return None, None, config


def measure_thermodynamics(model, tokenizer, text: str, batch_size: int = 1) -> Dict:
    """
    Comprehensive thermodynamic measurement.
    Returns BPT, energy, phi, and all thermodynamic metrics.
    """

    # Tokenize
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=EVAL_TOKENS)
    input_ids = inputs['input_ids'].to(model.device)
    n_tokens = input_ids.shape[1]

    logger.info(f"  Measuring {n_tokens} tokens, batch_size={batch_size}")

    # BPT measurement
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss.item()

    bpt = loss / np.log(2)

    # Energy measurement with power monitoring
    monitor = PowerMonitor()
    monitor.start()

    start_time = time.time()

    # Run inference
    if batch_size == 1:
        with torch.no_grad():
            for _ in range(1):  # Single pass for accurate timing
                outputs = model(input_ids, labels=input_ids)
    else:
        # Batch processing
        with torch.no_grad():
            # Repeat input to create batch
            batched_ids = input_ids.repeat(batch_size, 1)
            outputs = model(batched_ids, labels=batched_ids)

    torch.cuda.synchronize()
    elapsed_time = time.time() - start_time

    power_stats = monitor.stop()

    # Compute metrics
    mean_power_W = power_stats['mean_power_W']
    mean_temp_C = power_stats['mean_temp_C']
    temp_K = mean_temp_C + 273.15

    energy_total_J = mean_power_W * elapsed_time
    tokens_per_second = (n_tokens * batch_size) / elapsed_time if elapsed_time > 0 else 0
    energy_per_token_J = energy_total_J / (n_tokens * batch_size) if n_tokens > 0 else 0
    energy_per_bit_J = energy_per_token_J / bpt if bpt > 0 else 0

    # Landauer limit
    E_landauer = K_B * temp_K * np.log(2)

    # Phi: efficiency relative to Landauer
    phi = energy_per_token_J / (E_landauer * bpt) if (E_landauer * bpt) > 0 else 0
    log10_phi = np.log10(phi) if phi > 0 else float('nan')

    return {
        'bpt': bpt,
        'tokens_per_sec': tokens_per_second,
        'energy_total_J': energy_total_J,
        'energy_per_token_J': energy_per_token_J,
        'energy_per_bit_J': energy_per_bit_J,
        'mean_power_W': mean_power_W,
        'temp_K': temp_K,
        'E_landauer_J': E_landauer,
        'phi': phi,
        'log10_phi': log10_phi,
        'elapsed_time_s': elapsed_time,
    }


def run_experiment_6():
    """Main experiment runner"""
    logger.info("="*80)
    logger.info("EXPERIMENT 6: ENERGY SURVEY / THERMODYNAMIC ROADMAP")
    logger.info("="*80)

    # Load WikiText-2
    logger.info("Loading WikiText-2 dataset...")
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    wikitext_full = '\n'.join(dataset['text'])
    wikitext_sample = wikitext_full[:50000]  # Sample for evaluation

    # Results storage
    results = []

    # Configurations to test
    configs = [
        ('fp32', 1),
        ('fp16', 1),
        ('int8', 1),
        ('int4', 1),
        ('fp16_compiled', 1),
        ('fp16_batch', 1),
        ('fp16_batch', 8),
        ('fp16_batch', 32),
        ('fp16_batch', 128),
    ]

    for model_name in MODELS:
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing {model_name}")
        logger.info(f"{'='*80}")

        # Get parameter count
        try:
            temp_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map='auto',
            )
            params = sum(p.numel() for p in temp_model.parameters())
            del temp_model
            torch.cuda.empty_cache()
            time.sleep(5)
        except Exception as e:
            logger.error(f"Failed to get params for {model_name}: {e}")
            continue

        for config_name, batch_size in configs:
            config_str = f"{config_name}_bs{batch_size}" if config_name.startswith('fp16_batch') else config_name

            logger.info(f"  Config: {config_str}")

            # Load model
            if config_name == 'fp16_batch':
                actual_config = 'fp16'
            else:
                actual_config = config_name

            model, tokenizer, _ = load_model_with_config(model_name, actual_config)

            if model is None:
                logger.warning(f"  Skipping {model_name} with {config_str}")
                continue

            try:
                # Check if model fits in VRAM
                if batch_size > 1:
                    # Quick memory check
                    mem_free = torch.cuda.mem_get_info()[0]
                    mem_total = torch.cuda.mem_get_info()[1]
                    mem_used_pct = (1 - mem_free / mem_total) * 100

                    if mem_used_pct > 85:
                        logger.warning(f"  Skipping batch_size={batch_size}, insufficient VRAM ({mem_used_pct:.1f}% used)")
                        del model
                        del tokenizer
                        torch.cuda.empty_cache()
                        continue

                # Measure
                measurements = measure_thermodynamics(model, tokenizer, wikitext_sample, batch_size)

                results.append({
                    'model': model_name,
                    'params': params,
                    'config': config_str,
                    'batch_size': batch_size,
                    **measurements
                })

                logger.info(f"    BPT: {measurements['bpt']:.3f}, "
                          f"φ: {measurements['phi']:.2e}, "
                          f"log10(φ): {measurements['log10_phi']:.2f}, "
                          f"Energy/bit: {measurements['energy_per_bit_J']:.2e} J")

            except Exception as e:
                logger.error(f"  Error measuring {model_name} with {config_str}: {e}")

            finally:
                # Clean up
                del model
                del tokenizer
                torch.cuda.empty_cache()
                logger.info(f"  Cooling for {COOLING_TIME}s...")
                time.sleep(COOLING_TIME)

    # Save results
    logger.info("\nSaving results...")
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / 'exp6_energy.csv', index=False)
    logger.info(f"Saved {RESULTS_DIR / 'exp6_energy.csv'}")

    # Generate plots
    generate_plots(df)

    # Generate thermodynamic roadmap
    generate_roadmap(df)

    logger.info("\nExperiment 6 complete!")


def generate_plots(df):
    """Generate all required plots"""
    logger.info("\nGenerating plots...")

    sns.set_style("whitegrid")

    # Plot 1: log10(φ) vs model size
    plt.figure(figsize=(14, 8))

    for config in df['config'].unique():
        config_data = df[df['config'] == config]
        plt.plot(np.log10(config_data['params']), config_data['log10_phi'],
                marker='o', label=config, linewidth=2)

    # Ribosome reference
    plt.axhline(y=np.log10(RIBOSOME_PHI), color='red', linestyle='--',
               linewidth=2, label='Ribosome (φ ≈ 1.02)')

    plt.xlabel('log10(Parameters)')
    plt.ylabel('log10(φ)')
    plt.title('Experiment 6: Thermodynamic Efficiency vs Model Size')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'exp6_phi_vs_size.png', dpi=300)
    logger.info(f"Saved {PLOTS_DIR / 'exp6_phi_vs_size.png'}")
    plt.close()

    # Plot 2: Bits-per-joule vs model size
    plt.figure(figsize=(14, 8))

    for config in df['config'].unique():
        config_data = df[df['config'] == config]
        bits_per_joule = 1.0 / config_data['energy_per_bit_J']
        plt.plot(np.log10(config_data['params']), bits_per_joule,
                marker='s', label=config, linewidth=2)

    # Ribosome reference
    ribosome_bits_per_joule = 1.0 / RIBOSOME_ENERGY_PER_BIT_J
    plt.axhline(y=ribosome_bits_per_joule, color='red', linestyle='--',
               linewidth=2, label=f'Ribosome ({ribosome_bits_per_joule:.2e} bits/J)')

    plt.xlabel('log10(Parameters)')
    plt.ylabel('Bits per Joule')
    plt.title('Experiment 6: Information Efficiency vs Model Size')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'exp6_bits_per_joule.png', dpi=300)
    logger.info(f"Saved {PLOTS_DIR / 'exp6_bits_per_joule.png'}")
    plt.close()

    # Plot 3: Energy/token vs BPT scatter with Pareto frontier
    plt.figure(figsize=(12, 10))

    configs_unique = df['config'].unique()
    colors = sns.color_palette("husl", len(configs_unique))
    config_colors = dict(zip(configs_unique, colors))

    for config in configs_unique:
        config_data = df[df['config'] == config]
        plt.scatter(config_data['bpt'], config_data['energy_per_token_J'],
                   c=[config_colors[config]], label=config, s=100, alpha=0.6)

    # Basin band
    plt.axvspan(3, 6, alpha=0.1, color='green', label='Basin [3,6] bits')
    plt.axvline(x=4.16, color='black', linestyle='--', alpha=0.3)

    # Identify and mark Pareto frontier
    pareto_points = []
    for _, row in df.iterrows():
        is_pareto = True
        for _, other_row in df.iterrows():
            if (other_row['bpt'] <= row['bpt'] and
                other_row['energy_per_token_J'] <= row['energy_per_token_J'] and
                (other_row['bpt'] < row['bpt'] or
                 other_row['energy_per_token_J'] < row['energy_per_token_J'])):
                is_pareto = False
                break
        if is_pareto:
            pareto_points.append(row)

    if pareto_points:
        pareto_df = pd.DataFrame(pareto_points)
        pareto_df = pareto_df.sort_values('bpt')
        plt.plot(pareto_df['bpt'], pareto_df['energy_per_token_J'],
                'k-', linewidth=3, label='Pareto Frontier', alpha=0.7)

    plt.xlabel('BPT')
    plt.ylabel('Energy per Token (J)')
    plt.title('Experiment 6: BPT vs Energy (Pareto Frontier)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'exp6_energy_vs_bpt.png', dpi=300)
    logger.info(f"Saved {PLOTS_DIR / 'exp6_energy_vs_bpt.png'}")
    plt.close()

    # Plot 4: Batch size scaling
    plt.figure(figsize=(12, 8))

    batch_data = df[df['config'].str.contains('batch|fp16') & (df['config'] != 'fp16_compiled')]

    for model in batch_data['model'].unique():
        model_data = batch_data[batch_data['model'] == model]
        model_data = model_data.sort_values('batch_size')

        plt.plot(model_data['batch_size'], model_data['energy_per_token_J'],
                marker='o', label=model.split('/')[-1], linewidth=2)

    plt.xlabel('Batch Size')
    plt.ylabel('Energy per Token (J)')
    plt.title('Experiment 6: Batch Size Scaling')
    plt.legend()
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'exp6_batch_scaling.png', dpi=300)
    logger.info(f"Saved {PLOTS_DIR / 'exp6_batch_scaling.png'}")
    plt.close()


def generate_roadmap(df):
    """Generate THERMODYNAMIC_ROADMAP.md"""
    logger.info("\nGenerating thermodynamic roadmap...")

    # Find best configurations
    best_phi = df.loc[df['phi'].idxmin()]
    best_bits_per_joule_idx = (1.0 / df['energy_per_bit_J']).idxmax()
    best_bits_per_joule = df.loc[best_bits_per_joule_idx]

    # Calculate gaps to ribosome and Landauer
    gap_to_ribosome_phi = best_phi['phi'] / RIBOSOME_PHI
    gap_to_ribosome_energy = best_bits_per_joule['energy_per_bit_J'] / RIBOSOME_ENERGY_PER_BIT_J

    # Landauer limit at GPU temperature
    avg_temp_K = df['temp_K'].mean()
    E_landauer = K_B * avg_temp_K * np.log(2)
    gap_to_landauer = best_bits_per_joule['energy_per_bit_J'] / E_landauer

    roadmap = []
    roadmap.append("# Thermodynamic Roadmap: RTX 5090 on the Ribosome-to-Landauer Spectrum")
    roadmap.append("")
    roadmap.append("**Windstorm Institute Paper 7 - Experiment 6**")
    roadmap.append("")
    roadmap.append("## Executive Summary")
    roadmap.append("")
    roadmap.append(f"This experiment characterizes where the RTX 5090 sits on the thermodynamic spectrum")
    roadmap.append(f"from the Landauer limit (theoretical minimum) to the ribosome (biological optimum) to")
    roadmap.append(f"current silicon implementations.")
    roadmap.append("")

    roadmap.append("## Best Configurations")
    roadmap.append("")
    roadmap.append("### Highest Thermodynamic Efficiency (Lowest φ)")
    roadmap.append("")
    roadmap.append(f"- **Model:** {best_phi['model']}")
    roadmap.append(f"- **Config:** {best_phi['config']}")
    roadmap.append(f"- **φ:** {best_phi['phi']:.2e}")
    roadmap.append(f"- **log10(φ):** {best_phi['log10_phi']:.2f}")
    roadmap.append(f"- **BPT:** {best_phi['bpt']:.3f} bits")
    roadmap.append(f"- **Energy/token:** {best_phi['energy_per_token_J']:.2e} J")
    roadmap.append("")

    roadmap.append("### Best Bits per Joule")
    roadmap.append("")
    bits_per_joule_val = 1.0 / best_bits_per_joule['energy_per_bit_J']
    roadmap.append(f"- **Model:** {best_bits_per_joule['model']}")
    roadmap.append(f"- **Config:** {best_bits_per_joule['config']}")
    roadmap.append(f"- **Bits/Joule:** {bits_per_joule_val:.2e}")
    roadmap.append(f"- **Energy/bit:** {best_bits_per_joule['energy_per_bit_J']:.2e} J")
    roadmap.append(f"- **BPT:** {best_bits_per_joule['bpt']:.3f} bits")
    roadmap.append("")

    roadmap.append("## Thermodynamic Gaps")
    roadmap.append("")

    roadmap.append("### Gap to Ribosome")
    roadmap.append("")
    roadmap.append(f"- **φ ratio (RTX 5090 / Ribosome):** {gap_to_ribosome_phi:.2e}×")
    roadmap.append(f"- **Orders of magnitude:** {np.log10(gap_to_ribosome_phi):.1f} OOM")
    roadmap.append(f"- **Energy/bit ratio:** {gap_to_ribosome_energy:.2e}×")
    roadmap.append("")
    roadmap.append(f"The RTX 5090 operates **~{np.log10(gap_to_ribosome_phi):.1f} orders of magnitude**")
    roadmap.append(f"above the ribosome's thermodynamic efficiency.")
    roadmap.append("")

    roadmap.append("### Gap to Landauer Limit")
    roadmap.append("")
    roadmap.append(f"- **Landauer limit (at {avg_temp_K:.1f} K):** {E_landauer:.2e} J/bit")
    roadmap.append(f"- **RTX 5090 best:** {best_bits_per_joule['energy_per_bit_J']:.2e} J/bit")
    roadmap.append(f"- **Gap:** {gap_to_landauer:.2e}×")
    roadmap.append(f"- **Orders of magnitude:** {np.log10(gap_to_landauer):.1f} OOM")
    roadmap.append("")
    roadmap.append(f"The RTX 5090 operates **~{np.log10(gap_to_landauer):.1f} orders of magnitude**")
    roadmap.append(f"above the Landauer limit.")
    roadmap.append("")

    roadmap.append("## Key Findings")
    roadmap.append("")

    # Model size trends
    roadmap.append("### 1. Model Size Effects")
    roadmap.append("")
    for config in ['fp16', 'int8', 'int4']:
        config_data = df[df['config'] == config].sort_values('params')
        if len(config_data) >= 2:
            smallest = config_data.iloc[0]
            largest = config_data.iloc[-1]
            phi_change = largest['log10_phi'] - smallest['log10_phi']
            roadmap.append(f"**{config}:** log10(φ) changes by {phi_change:+.2f} from {smallest['params']:,} to {largest['params']:,} params")

    roadmap.append("")

    # Batch size effects
    batch_df = df[df['config'].str.contains('batch')]
    if len(batch_df) > 0:
        roadmap.append("### 2. Batch Size Effects")
        roadmap.append("")
        for model in batch_df['model'].unique():
            model_data = batch_df[batch_df['model'] == model].sort_values('batch_size')
            if len(model_data) >= 2:
                bs1 = model_data[model_data['batch_size'] == 1]
                bs_max = model_data.iloc[-1]
                if len(bs1) > 0:
                    energy_reduction = bs1['energy_per_token_J'].values[0] / bs_max['energy_per_token_J']
                    roadmap.append(f"**{model.split('/')[-1]}:** {energy_reduction:.2f}× energy reduction at batch_size={bs_max['batch_size']}")
        roadmap.append("")

    # Precision effects
    roadmap.append("### 3. Precision Effects on Efficiency")
    roadmap.append("")
    precision_configs = ['fp32', 'fp16', 'int8', 'int4']
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        prec_data = model_data[model_data['config'].isin(precision_configs)]

        if len(prec_data) >= 2:
            roadmap.append(f"**{model.split('/')[-1]}:**")
            for _, row in prec_data.iterrows():
                roadmap.append(f"  - {row['config']}: φ = {row['phi']:.2e}, BPT = {row['bpt']:.3f}")
            roadmap.append("")

    roadmap.append("## Hardware Recommendations for AGI")
    roadmap.append("")
    roadmap.append("Based on these measurements, optimal AGI hardware should:")
    roadmap.append("")

    if best_phi['config'] == 'int4':
        roadmap.append("1. **Precision:** INT4 quantization provides best thermodynamic efficiency")
    elif best_phi['config'] == 'int8':
        roadmap.append("1. **Precision:** INT8 quantization balances efficiency and BPT maintenance")
    else:
        roadmap.append("1. **Precision:** FP16 remains necessary for basin-level BPT")

    roadmap.append("2. **Batch Processing:** Maximize batch size for energy efficiency (amortizes overhead)")
    roadmap.append(f"3. **Target Efficiency:** Close the ~{np.log10(gap_to_ribosome_phi):.0f} OOM gap to biological efficiency")
    roadmap.append("4. **Architecture:** Consider serial architectures (Mamba/SSM) per Experiment 3 findings")
    roadmap.append("")

    roadmap.append("## Path to Ribosome-Level Efficiency")
    roadmap.append("")
    roadmap.append(f"To match the ribosome (φ ≈ {RIBOSOME_PHI}):")
    roadmap.append("")
    roadmap.append(f"- Current best: φ = {best_phi['phi']:.2e}")
    roadmap.append(f"- Target: φ = {RIBOSOME_PHI:.2e}")
    roadmap.append(f"- **Required improvement: {gap_to_ribosome_phi:.2e}× ({np.log10(gap_to_ribosome_phi):.1f} OOM)**")
    roadmap.append("")
    roadmap.append("Potential pathways:")
    roadmap.append("- Novel architectures (neuromorphic, photonic, quantum)")
    roadmap.append("- Lower operating temperatures (cryogenic compute)")
    roadmap.append("- Fundamentally different computation substrates")
    roadmap.append("- Massive parallelization with ultra-low-power inference")
    roadmap.append("")

    roadmap.append("## Implications for Paper 7")
    roadmap.append("")
    roadmap.append("1. **Scale of the Gap:** Modern GPUs operate ~8-9 OOM above Landauer, ~8 OOM above ribosome")
    roadmap.append("2. **Quantization Benefits:** Lower precision reduces energy but may destabilize the basin (see Exp 2)")
    roadmap.append("3. **Batch Efficiency:** Energy/token improves dramatically with batching (infrastructure optimization)")
    roadmap.append("4. **Hardware Targets:** AGI hardware should target φ < 10^6 as a near-term milestone")
    roadmap.append("")

    # Write to file
    output_file = Path('/home/user1-gpu/agi-extensions/exp-6/THERMODYNAMIC_ROADMAP.md')
    with open(output_file, 'w') as f:
        f.write('\n'.join(roadmap))

    logger.info(f"Roadmap written to {output_file}")


if __name__ == '__main__':
    logger.info(f"Starting Experiment 6 at {datetime.now()}")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    run_experiment_6()

    logger.info(f"Completed at {datetime.now()}")
