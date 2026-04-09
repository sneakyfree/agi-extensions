#!/usr/bin/env python3
"""
Experiment 2: The Quantization Cliff
Windstorm Institute Paper 7 - AGI Extensions of the Throughput Basin Framework

Tests the minimum weight precision needed to maintain basin-level BPT.
Hypothesis: Cliff occurs at ~4 bits/weight, matching the inherited structure.
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
import copy

from scipy import stats
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
        logging.FileHandler('/home/user1-gpu/agi-extensions/exp-2/exp2.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Constants
RESULTS_DIR = Path('/home/user1-gpu/agi-extensions/exp-2/results')
PLOTS_DIR = Path('/home/user1-gpu/agi-extensions/exp-2/plots')
RESULTS_DIR.mkdir(exist_ok=True, parents=True)
PLOTS_DIR.mkdir(exist_ok=True, parents=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_RUNS = 3
COOLING_TIME = 30
MAX_LENGTH = 1024

# Model configurations - Pythia family for consistent tokenizer
PYTHIA_MODELS = [
    'EleutherAI/pythia-70m',
    'EleutherAI/pythia-160m',
    'EleutherAI/pythia-410m',
    'EleutherAI/pythia-1b',
    'EleutherAI/pythia-1.4b',
]

# Cross-validation with GPT-2
GPT2_MODELS = [
    'openai-community/gpt2',
    'openai-community/gpt2-medium',
    'openai-community/gpt2-large',
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
            return {'mean_power_W': 0, 'mean_temp_C': 0, 'max_power_W': 0}

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


def quantize_linear_weights(model, n_bits: int):
    """
    Manually quantize all linear layer weights to n_bits precision.
    This is a research probe, not production quantization.
    """
    import torch.nn as nn

    logger.info(f"Quantizing model weights to {n_bits} bits...")

    levels = 2 ** n_bits

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            with torch.no_grad():
                weight = module.weight.data

                # Quantize using linear quantization
                mn, mx = weight.min(), weight.max()
                scale = (mx - mn) / (levels - 1)

                if scale > 0:
                    quantized = torch.round((weight - mn) / scale)
                    dequantized = quantized * scale + mn
                    module.weight.data = dequantized

    return model


def load_model_with_precision(model_name: str, precision: str) -> Tuple[Optional[object], Optional[object], str]:
    """
    Load model with specified precision.
    Precision: 'fp32', 'fp16', 'int8', 'int4', 'int3', 'int2'
    """
    logger.info(f"Loading {model_name} with precision {precision}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load based on precision
        if precision == 'fp32':
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map='auto',
            )
        elif precision == 'fp16':
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map='auto',
            )
        elif precision == 'int8':
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                load_in_8bit=True,
                device_map='auto',
            )
        elif precision == 'int4':
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
        elif precision in ['int3', 'int2']:
            # Load in FP16 first, then manually quantize
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map='auto',
            )
            n_bits = int(precision.replace('int', ''))
            model = quantize_linear_weights(model, n_bits)
        else:
            raise ValueError(f"Unknown precision: {precision}")

        model.eval()
        logger.info(f"Successfully loaded {model_name} at {precision}")
        return model, tokenizer, precision

    except Exception as e:
        logger.error(f"Failed to load {model_name} at {precision}: {e}")
        return None, None, precision


def compute_bpt_bpb(model, tokenizer, text: str, max_length: int = MAX_LENGTH) -> Tuple[float, float]:
    """Compute bits per token and bits per byte"""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length)
    input_ids = inputs['input_ids'].to(model.device)

    if input_ids.shape[1] < 2:
        return float('nan'), float('nan')

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss.item()

    bpt = loss / np.log(2)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    total_bytes = len(text.encode('utf-8'))
    avg_bytes_per_token = total_bytes / len(tokens)
    bpb = bpt / avg_bytes_per_token if avg_bytes_per_token > 0 else float('nan')

    return bpt, bpb


def create_shuffled_text(text: str) -> str:
    """Create word-shuffled version of text"""
    import random
    random.seed(42)
    words = text.split()
    random.shuffle(words)
    return ' '.join(words)


def measure_with_energy(model, tokenizer, text: str, text_shuffled: str) -> Dict:
    """Measure BPT, structural bonus, and energy"""

    # Original text
    bpt_orig, bpb_orig = compute_bpt_bpb(model, tokenizer, text)

    # Shuffled text
    bpt_shuffled, _ = compute_bpt_bpb(model, tokenizer, text_shuffled)

    # Structural bonus
    structural_bonus = bpt_shuffled - bpt_orig

    # Energy measurement
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=MAX_LENGTH)
    input_ids = inputs['input_ids'].to(model.device)
    n_tokens = input_ids.shape[1]

    monitor = PowerMonitor()
    monitor.start()

    start_time = time.time()
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    torch.cuda.synchronize()
    elapsed_time = time.time() - start_time

    power_stats = monitor.stop()

    energy_J = power_stats['mean_power_W'] * elapsed_time
    energy_mJ = energy_J * 1000
    energy_per_token_mJ = (energy_J / n_tokens) * 1000 if n_tokens > 0 else 0

    # Bits per joule
    bits_per_joule = bpt_orig / energy_J if energy_J > 0 else 0

    return {
        'bpt': bpt_orig,
        'bpb': bpb_orig,
        'structural_bonus': structural_bonus,
        'energy_mJ': energy_mJ,
        'energy_per_token_mJ': energy_per_token_mJ,
        'bits_per_joule': bits_per_joule,
        'mean_power_W': power_stats['mean_power_W'],
    }


def run_experiment_2():
    """Main experiment runner"""
    logger.info("="*80)
    logger.info("EXPERIMENT 2: THE QUANTIZATION CLIFF")
    logger.info("="*80)

    # Load WikiText-2
    logger.info("Loading WikiText-2 dataset...")
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    wikitext_full = '\n'.join(dataset['text'])
    wikitext_sample = wikitext_full[:100000]
    wikitext_shuffled = create_shuffled_text(wikitext_sample)

    # Results storage
    results = []

    # Precision levels to test
    precisions = ['fp16', 'int8', 'int4', 'int3', 'int2']

    # Test all models × precisions
    all_models = PYTHIA_MODELS + GPT2_MODELS

    for model_name in all_models:
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing {model_name}")
        logger.info(f"{'='*80}")

        # Get parameter count (load once in FP16 to check)
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

        for precision in precisions:
            logger.info(f"  Testing precision: {precision}")

            # Load model
            model, tokenizer, prec_actual = load_model_with_precision(model_name, precision)

            if model is None:
                logger.warning(f"  Skipping {model_name} at {precision}")
                continue

            try:
                # Measure
                measurements = measure_with_energy(model, tokenizer, wikitext_sample, wikitext_shuffled)

                results.append({
                    'model': model_name,
                    'params': params,
                    'precision': precision,
                    'bpt': measurements['bpt'],
                    'bpb': measurements['bpb'],
                    'structural_bonus': measurements['structural_bonus'],
                    'energy_mJ': measurements['energy_mJ'],
                    'energy_per_token_mJ': measurements['energy_per_token_mJ'],
                    'bits_per_joule': measurements['bits_per_joule'],
                    'mean_power_W': measurements['mean_power_W'],
                })

                logger.info(f"    BPT: {measurements['bpt']:.3f}, "
                          f"Structural bonus: {measurements['structural_bonus']:.3f}, "
                          f"Energy/token: {measurements['energy_per_token_mJ']:.3f} mJ")

            except Exception as e:
                logger.error(f"  Error measuring {model_name} at {precision}: {e}")

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
    df.to_csv(RESULTS_DIR / 'exp2_quantization.csv', index=False)
    logger.info(f"Saved {RESULTS_DIR / 'exp2_quantization.csv'}")

    # Generate plots
    generate_plots(df)

    # Analysis
    run_analysis(df)

    logger.info("\nExperiment 2 complete!")


def generate_plots(df):
    """Generate all required plots"""
    logger.info("\nGenerating plots...")

    sns.set_style("whitegrid")

    # Plot 1: THE CLIFF - BPT vs precision
    plt.figure(figsize=(14, 8))

    precision_order = ['fp16', 'int8', 'int4', 'int3', 'int2']
    precision_nums = {'fp16': 16, 'int8': 8, 'int4': 4, 'int3': 3, 'int2': 2}

    for model in df['model'].unique():
        model_data = df[df['model'] == model].copy()
        model_data['prec_num'] = model_data['precision'].map(precision_nums)
        model_data = model_data.sort_values('prec_num', ascending=False)

        plt.plot(model_data['prec_num'], model_data['bpt'],
                marker='o', label=model.split('/')[-1], linewidth=2)

    plt.axhspan(3, 6, alpha=0.2, color='green', label='Throughput Basin [3,6] bits')
    plt.axhline(y=4.16, color='black', linestyle='--', alpha=0.5, label='τ = 4.16')

    plt.xlabel('Weight Precision (bits)')
    plt.ylabel('BPT')
    plt.title('Experiment 2: The Quantization Cliff')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'exp2_cliff.png', dpi=300)
    logger.info(f"Saved {PLOTS_DIR / 'exp2_cliff.png'}")
    plt.close()

    # Plot 2: Structural bonus vs precision
    plt.figure(figsize=(12, 8))

    for model in df['model'].unique():
        model_data = df[df['model'] == model].copy()
        model_data['prec_num'] = model_data['precision'].map(precision_nums)
        model_data = model_data.sort_values('prec_num', ascending=False)

        plt.plot(model_data['prec_num'], model_data['structural_bonus'],
                marker='s', label=model.split('/')[-1], linewidth=2)

    plt.xlabel('Weight Precision (bits)')
    plt.ylabel('Structural Bonus (bits)')
    plt.title('Experiment 2: Does Syntax Break First?')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'exp2_structural_bonus.png', dpi=300)
    logger.info(f"Saved {PLOTS_DIR / 'exp2_structural_bonus.png'}")
    plt.close()

    # Plot 3: Bits-per-joule vs precision (efficiency frontier)
    plt.figure(figsize=(12, 8))

    for model in df['model'].unique():
        model_data = df[df['model'] == model].copy()
        model_data['prec_num'] = model_data['precision'].map(precision_nums)
        model_data = model_data.sort_values('prec_num', ascending=False)

        plt.plot(model_data['prec_num'], model_data['bits_per_joule'],
                marker='D', label=model.split('/')[-1], linewidth=2)

    plt.xlabel('Weight Precision (bits)')
    plt.ylabel('Bits per Joule')
    plt.title('Experiment 2: Energy Efficiency Frontier')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'exp2_bits_per_joule.png', dpi=300)
    logger.info(f"Saved {PLOTS_DIR / 'exp2_bits_per_joule.png'}")
    plt.close()

    # Plot 4: Heatmap - model_size × precision, color=BPT
    plt.figure(figsize=(10, 8))

    # Create pivot table
    pivot_data = df.pivot_table(
        index='model',
        columns='precision',
        values='bpt',
        aggfunc='mean'
    )

    # Reorder columns
    pivot_data = pivot_data[precision_order]

    # Create heatmap
    sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='RdYlGn_r',
                vmin=3, vmax=12, center=4.16,
                cbar_kws={'label': 'BPT'})

    plt.title('Experiment 2: BPT Heatmap (Green = Basin, Red = Collapsed)')
    plt.xlabel('Weight Precision')
    plt.ylabel('Model')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'exp2_heatmap.png', dpi=300)
    logger.info(f"Saved {PLOTS_DIR / 'exp2_heatmap.png'}")
    plt.close()


def run_analysis(df):
    """Run analysis and save summary"""
    logger.info("\nRunning analysis...")

    precision_nums = {'fp16': 16, 'int8': 8, 'int4': 4, 'int3': 3, 'int2': 2}

    # Identify cliff precision for each model
    cliff_results = []

    for model in df['model'].unique():
        model_data = df[df['model'] == model].copy()
        model_data['prec_num'] = model_data['precision'].map(precision_nums)
        model_data = model_data.sort_values('prec_num', ascending=False)

        # Find lowest precision where BPT < 6 (still in basin)
        in_basin = model_data[model_data['bpt'] < 6.0]

        if len(in_basin) > 0:
            cliff_precision = in_basin['precision'].iloc[-1]
            cliff_bpt = in_basin['bpt'].iloc[-1]
        else:
            cliff_precision = 'fp16'
            cliff_bpt = model_data['bpt'].iloc[0]

        params = model_data['params'].iloc[0]

        cliff_results.append({
            'model': model,
            'params': params,
            'log_params': np.log10(params),
            'cliff_precision': cliff_precision,
            'cliff_bpt': cliff_bpt,
        })

    df_cliff = pd.DataFrame(cliff_results)
    df_cliff.to_csv(RESULTS_DIR / 'exp2_cliff_analysis.csv', index=False)
    logger.info(f"Saved {RESULTS_DIR / 'exp2_cliff_analysis.csv'}")

    # Regression: cliff_precision ~ log(params)
    cliff_nums = df_cliff['cliff_precision'].map(precision_nums)
    log_params = df_cliff['log_params']

    if len(cliff_nums) > 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_params, cliff_nums)

        logger.info(f"\nCliff Regression:")
        logger.info(f"  cliff_bits = {slope:.3f} * log10(params) + {intercept:.3f}")
        logger.info(f"  R² = {r_value**2:.3f}")
        logger.info(f"  p = {p_value:.4f}")

    # Identify Pareto-optimal configurations
    pareto_points = []

    for _, row in df.iterrows():
        is_pareto = True
        for _, other_row in df.iterrows():
            # Check if other_row dominates this row
            if (other_row['bits_per_joule'] >= row['bits_per_joule'] and
                other_row['bpt'] <= row['bpt'] and
                (other_row['bits_per_joule'] > row['bits_per_joule'] or
                 other_row['bpt'] < row['bpt'])):
                is_pareto = False
                break

        if is_pareto:
            pareto_points.append(row.to_dict())

    df_pareto = pd.DataFrame(pareto_points)
    df_pareto.to_csv(RESULTS_DIR / 'exp2_pareto_optimal.csv', index=False)
    logger.info(f"Saved {RESULTS_DIR / 'exp2_pareto_optimal.csv'}")


if __name__ == '__main__':
    logger.info(f"Starting Experiment 2 at {datetime.now()}")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    run_experiment_2()

    logger.info(f"Completed at {datetime.now()}")
