#!/usr/bin/env python3
"""
Experiment 3: Recurrent vs Transformer Architecture Comparison
Windstorm Institute Paper 7 - AGI Extensions of the Throughput Basin Framework

Tests whether serial architectures (Mamba/RWKV) show tighter convergence to τ=4.16 bits
than transformers, which would indicate architectural vs. data-driven basin constraints.
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

# Scientific libraries
from scipy import stats
from datasets import load_dataset

# Transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/user1-gpu/agi-extensions/exp-3/exp3.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Constants
RESULTS_DIR = Path('/home/user1-gpu/agi-extensions/exp-3/results')
PLOTS_DIR = Path('/home/user1-gpu/agi-extensions/exp-3/plots')
RESULTS_DIR.mkdir(exist_ok=True, parents=True)
PLOTS_DIR.mkdir(exist_ok=True, parents=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_RUNS = 3
COOLING_TIME = 30  # seconds between runs
MAX_LENGTH = 1024
BATCH_SIZE = 1

# Model configurations
TRANSFORMER_MODELS = [
    'EleutherAI/pythia-160m',
    'EleutherAI/pythia-410m',
    'EleutherAI/pythia-1.4b',
    'openai-community/gpt2-medium',
]

SERIAL_MODELS = [
    'state-spaces/mamba-130m-hf',
    'state-spaces/mamba-370m-hf',
    'state-spaces/mamba-1.4b-hf',
]

# Try RWKV if available (optional)
RWKV_MODELS = [
    'RWKV/rwkv-4-430m-pile',
]


class PowerMonitor:
    """Background power monitoring using nvidia-smi"""

    def __init__(self):
        self.power_readings = []
        self.temp_readings = []
        self.running = False
        self.thread = None

    def start(self):
        """Start monitoring in background thread"""
        self.power_readings = []
        self.temp_readings = []
        self.running = True
        self.thread = threading.Thread(target=self._monitor)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        """Stop monitoring and return statistics"""
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
        """Monitor loop - runs in background thread"""
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
            time.sleep(0.1)  # 10 Hz sampling


def load_model_safe(model_name: str, arch_type: str) -> Tuple[Optional[object], Optional[object], Dict]:
    """
    Safely load model and tokenizer with error handling.
    Returns (model, tokenizer, metadata) or (None, None, {}) on failure.
    """
    logger.info(f"Loading {model_name} ({arch_type})")
    metadata = {'model_name': model_name, 'arch_type': arch_type}

    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map='auto',
            trust_remote_code=True,
        )
        model.eval()

        # Extract metadata
        metadata['params'] = sum(p.numel() for p in model.parameters())
        metadata['vocab_size'] = model.config.vocab_size

        logger.info(f"Successfully loaded {model_name}: {metadata['params']:,} params")
        return model, tokenizer, metadata

    except Exception as e:
        logger.error(f"Failed to load {model_name}: {e}")
        return None, None, metadata


def compute_bpt_bpb(model, tokenizer, text: str, max_length: int = MAX_LENGTH) -> Tuple[float, float]:
    """
    Compute bits per token (BPT) and bits per byte (BPB) via teacher-forced cross-entropy.
    BPT = loss / ln(2)
    BPB = BPT / avg_bytes_per_token
    """
    # Tokenize
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length)
    input_ids = inputs['input_ids'].to(model.device)

    if input_ids.shape[1] < 2:
        return float('nan'), float('nan')

    # Forward pass with labels
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss.item()

    # BPT = cross-entropy loss / ln(2)
    bpt = loss / np.log(2)

    # BPB: need to compute bytes per token
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    total_bytes = len(text.encode('utf-8'))
    avg_bytes_per_token = total_bytes / len(tokens)
    bpb = bpt / avg_bytes_per_token if avg_bytes_per_token > 0 else float('nan')

    return bpt, bpb


def measure_bpt_repeated(model, tokenizer, text: str, n_runs: int = N_RUNS) -> Dict:
    """Run BPT measurement n_runs times with cooling periods, return mean ± std"""
    bpt_values = []
    bpb_values = []

    for run in range(n_runs):
        if run > 0:
            logger.info(f"Cooling for {COOLING_TIME}s...")
            time.sleep(COOLING_TIME)

        logger.info(f"Run {run+1}/{n_runs}")
        bpt, bpb = compute_bpt_bpb(model, tokenizer, text)
        bpt_values.append(bpt)
        bpb_values.append(bpb)

    return {
        'bpt_mean': np.mean(bpt_values),
        'bpt_std': np.std(bpt_values),
        'bpb_mean': np.mean(bpb_values),
        'bpb_std': np.std(bpb_values),
    }


def create_shuffled_texts(text: str) -> Dict[str, str]:
    """
    Create 5 versions of text with increasing shuffling:
    1. Original
    2. Paragraphs shuffled
    3. Sentences shuffled (within paragraphs)
    4. Words shuffled (within sentences)
    5. All words shuffled globally
    """
    import random
    random.seed(42)

    versions = {'original': text}

    # Split into paragraphs
    paragraphs = text.split('\n\n')
    paragraphs = [p for p in paragraphs if p.strip()]

    # 2. Paragraphs shuffled
    shuffled_paras = paragraphs.copy()
    random.shuffle(shuffled_paras)
    versions['paragraphs_shuffled'] = '\n\n'.join(shuffled_paras)

    # 3. Sentences shuffled within paragraphs
    sentence_shuffled_paras = []
    for para in paragraphs:
        sentences = para.replace('!', '.').replace('?', '.').split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        random.shuffle(sentences)
        sentence_shuffled_paras.append('. '.join(sentences) + '.')
    versions['sentences_shuffled'] = '\n\n'.join(sentence_shuffled_paras)

    # 4. Words shuffled within sentences
    word_shuffled_paras = []
    for para in paragraphs:
        sentences = para.replace('!', '.').replace('?', '.').split('.')
        word_shuffled_sents = []
        for sent in sentences:
            words = sent.split()
            if words:
                random.shuffle(words)
                word_shuffled_sents.append(' '.join(words))
        word_shuffled_paras.append('. '.join(word_shuffled_sents) + '.')
    versions['words_shuffled'] = '\n\n'.join(word_shuffled_paras)

    # 5. All words shuffled globally
    all_words = text.split()
    random.shuffle(all_words)
    versions['all_shuffled'] = ' '.join(all_words)

    return versions


def generate_seven_corpora() -> Dict[str, str]:
    """Generate the seven test corpora"""
    import random
    random.seed(42)
    np.random.seed(42)

    corpora = {}

    # 1. WikiText-2 (loaded separately)
    corpora['wikitext'] = None  # Placeholder

    # 2. Python code
    python_samples = [
        "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)\n",
        "class Node:\n    def __init__(self, value):\n        self.value = value\n        self.next = None\n",
        "import numpy as np\ndef compute_mean(arr):\n    return np.mean(arr)\n",
        "for i in range(100):\n    if i % 2 == 0:\n        print(f'{i} is even')\n",
    ]
    corpora['python'] = '\n'.join(python_samples * 200)  # Repeat to get enough tokens

    # 3. DNA (random ACGT)
    dna_length = 50000
    corpora['dna'] = ''.join(random.choices('ACGT', k=dna_length))

    # 4. Shuffled WikiText-2 (will be created from actual wikitext)
    corpora['shuffled_wikitext'] = None  # Placeholder

    # 5. Math expressions
    math_samples = []
    for _ in range(1000):
        a, b = random.randint(1, 999), random.randint(1, 999)
        op = random.choice(['+', '-', '*'])
        if op == '+':
            result = a + b
        elif op == '-':
            result = a - b
        else:
            result = a * b
        math_samples.append(f"{a} {op} {b} = {result}")
    corpora['math'] = '\n'.join(math_samples)

    # 6. Random ASCII
    ascii_chars = [chr(i) for i in range(32, 127)]
    corpora['random_ascii'] = ''.join(random.choices(ascii_chars, k=50000))

    # 7. Synthetic CSV
    csv_lines = ["name,age,city"]
    names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry"]
    cities = ["NYC", "LA", "Chicago", "Houston", "Phoenix", "Philadelphia"]
    for _ in range(1000):
        name = random.choice(names)
        age = random.randint(20, 80)
        city = random.choice(cities)
        csv_lines.append(f"{name},{age},{city}")
    corpora['csv'] = '\n'.join(csv_lines)

    return corpora


def measure_energy_per_token(model, tokenizer, text: str) -> Dict:
    """Measure energy consumption during inference"""
    # Tokenize
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=MAX_LENGTH)
    input_ids = inputs['input_ids'].to(model.device)
    n_tokens = input_ids.shape[1]

    # Start power monitoring
    monitor = PowerMonitor()
    monitor.start()

    # Run inference
    start_time = time.time()
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    torch.cuda.synchronize()
    elapsed_time = time.time() - start_time

    # Stop monitoring
    power_stats = monitor.stop()

    # Compute energy
    energy_J = power_stats['mean_power_W'] * elapsed_time
    energy_per_token_mJ = (energy_J / n_tokens) * 1000 if n_tokens > 0 else 0

    return {
        'elapsed_time_s': elapsed_time,
        'n_tokens': n_tokens,
        'energy_total_J': energy_J,
        'energy_per_token_mJ': energy_per_token_mJ,
        **power_stats
    }


def run_experiment_3():
    """Main experiment runner"""
    logger.info("="*80)
    logger.info("EXPERIMENT 3: RECURRENT VS TRANSFORMER")
    logger.info("="*80)

    # Load WikiText-2 dataset
    logger.info("Loading WikiText-2 dataset...")
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    wikitext_full = '\n'.join(dataset['text'])
    wikitext_sample = wikitext_full[:100000]  # First 100k chars for testing

    # Create shuffled versions
    logger.info("Creating shuffled versions...")
    shuffled_versions = create_shuffled_texts(wikitext_sample)

    # Generate seven corpora
    logger.info("Generating seven corpora...")
    corpora = generate_seven_corpora()
    corpora['wikitext'] = wikitext_sample
    corpora['shuffled_wikitext'] = shuffled_versions['all_shuffled']

    # Results storage
    results_bpt = []
    results_shuffling = []
    results_seven_corpus = []
    results_energy = []

    # Test all models
    all_models = [
        *[(m, 'transformer') for m in TRANSFORMER_MODELS],
        *[(m, 'serial') for m in SERIAL_MODELS],
    ]

    for model_name, arch_type in all_models:
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing {model_name} ({arch_type})")
        logger.info(f"{'='*80}")

        # Load model
        model, tokenizer, metadata = load_model_safe(model_name, arch_type)
        if model is None:
            logger.warning(f"Skipping {model_name} - failed to load")
            continue

        try:
            # A) BPT and BPB on WikiText-2
            logger.info("A) Measuring BPT/BPB on WikiText-2...")
            bpt_results = measure_bpt_repeated(model, tokenizer, wikitext_sample)

            results_bpt.append({
                'model': model_name,
                'arch_type': arch_type,
                'params': metadata['params'],
                'vocab_size': metadata['vocab_size'],
                'bpt_mean': bpt_results['bpt_mean'],
                'bpt_std': bpt_results['bpt_std'],
                'bpb_mean': bpt_results['bpb_mean'],
                'bpb_std': bpt_results['bpb_std'],
            })

            # B) Shuffling cascade
            logger.info("B) Running shuffling cascade...")
            for shuffle_level, text in shuffled_versions.items():
                bpt, bpb = compute_bpt_bpb(model, tokenizer, text)
                delta = bpt - bpt_results['bpt_mean']

                results_shuffling.append({
                    'model': model_name,
                    'arch_type': arch_type,
                    'shuffle_level': shuffle_level,
                    'bpt': bpt,
                    'delta': delta,
                })

            # C) Seven-corpus battery
            logger.info("C) Running seven-corpus battery...")
            for corpus_name, corpus_text in corpora.items():
                if corpus_text is None:
                    continue
                bpt, _ = compute_bpt_bpb(model, tokenizer, corpus_text[:50000])

                results_seven_corpus.append({
                    'model': model_name,
                    'arch_type': arch_type,
                    'corpus': corpus_name,
                    'bpt': bpt,
                })

            # D) Energy per token
            logger.info("D) Measuring energy consumption...")
            energy_stats = measure_energy_per_token(model, tokenizer, wikitext_sample)

            results_energy.append({
                'model': model_name,
                'arch_type': arch_type,
                'energy_per_token_mJ': energy_stats['energy_per_token_mJ'],
                'mean_power_W': energy_stats['mean_power_W'],
                'mean_temp_C': energy_stats['mean_temp_C'],
            })

            logger.info(f"Completed {model_name}")

        except Exception as e:
            logger.error(f"Error testing {model_name}: {e}")

        finally:
            # Clean up
            del model
            del tokenizer
            torch.cuda.empty_cache()
            logger.info(f"Cooling for {COOLING_TIME}s...")
            time.sleep(COOLING_TIME)

    # Save results
    logger.info("\nSaving results...")

    df_bpt = pd.DataFrame(results_bpt)
    df_bpt.to_csv(RESULTS_DIR / 'exp3_bpt_comparison.csv', index=False)
    logger.info(f"Saved {RESULTS_DIR / 'exp3_bpt_comparison.csv'}")

    df_shuffling = pd.DataFrame(results_shuffling)
    df_shuffling.to_csv(RESULTS_DIR / 'exp3_shuffling_cascade.csv', index=False)
    logger.info(f"Saved {RESULTS_DIR / 'exp3_shuffling_cascade.csv'}")

    df_seven = pd.DataFrame(results_seven_corpus)
    df_seven.to_csv(RESULTS_DIR / 'exp3_seven_corpus.csv', index=False)
    logger.info(f"Saved {RESULTS_DIR / 'exp3_seven_corpus.csv'}")

    df_energy = pd.DataFrame(results_energy)
    df_energy.to_csv(RESULTS_DIR / 'exp3_energy.csv', index=False)
    logger.info(f"Saved {RESULTS_DIR / 'exp3_energy.csv'}")

    # Generate plots
    generate_plots(df_bpt, df_shuffling, df_seven, df_energy)

    # Statistical analysis
    run_statistical_analysis(df_bpt, df_shuffling, df_seven)

    logger.info("\nExperiment 3 complete!")


def generate_plots(df_bpt, df_shuffling, df_seven, df_energy):
    """Generate all required plots"""
    logger.info("\nGenerating plots...")

    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)

    # Plot 1: BPT by model
    plt.figure(figsize=(14, 8))
    colors = {'transformer': 'blue', 'serial': 'red'}
    df_bpt_sorted = df_bpt.sort_values('bpt_mean')

    bars = plt.bar(range(len(df_bpt_sorted)), df_bpt_sorted['bpt_mean'],
                   yerr=df_bpt_sorted['bpt_std'], capsize=5,
                   color=[colors[arch] for arch in df_bpt_sorted['arch_type']])

    plt.axhspan(3, 6, alpha=0.2, color='green', label='Throughput Basin [3,6] bits')
    plt.axhline(y=4.16, color='black', linestyle='--', label='τ = 4.16 bits')

    plt.xticks(range(len(df_bpt_sorted)),
               [m.split('/')[-1] for m in df_bpt_sorted['model']],
               rotation=45, ha='right')
    plt.ylabel('Bits Per Token (BPT)')
    plt.title('Experiment 3: BPT by Model Architecture')
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'exp3_bpt_comparison.png', dpi=300)
    logger.info(f"Saved {PLOTS_DIR / 'exp3_bpt_comparison.png'}")
    plt.close()

    # Plot 2: Shuffling cascade
    plt.figure(figsize=(12, 8))
    shuffle_order = ['original', 'paragraphs_shuffled', 'sentences_shuffled',
                     'words_shuffled', 'all_shuffled']

    for model in df_shuffling['model'].unique():
        model_data = df_shuffling[df_shuffling['model'] == model]
        arch_type = model_data['arch_type'].iloc[0]

        ordered_data = []
        for level in shuffle_order:
            row = model_data[model_data['shuffle_level'] == level]
            if not row.empty:
                ordered_data.append(row['bpt'].values[0])
            else:
                ordered_data.append(np.nan)

        color = 'blue' if arch_type == 'transformer' else 'red'
        plt.plot(shuffle_order, ordered_data, marker='o', label=model.split('/')[-1],
                color=color, alpha=0.7)

    plt.xlabel('Shuffle Level')
    plt.ylabel('BPT')
    plt.title('Experiment 3: Shuffling Cascade')
    plt.xticks(rotation=45, ha='right')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'exp3_shuffling_cascade.png', dpi=300)
    logger.info(f"Saved {PLOTS_DIR / 'exp3_shuffling_cascade.png'}")
    plt.close()

    # Plot 3: BPT variance across 7 corpora - boxplot
    plt.figure(figsize=(10, 8))

    transformer_data = []
    serial_data = []

    for arch_type in df_seven['arch_type'].unique():
        arch_data = df_seven[df_seven['arch_type'] == arch_type]
        if arch_type == 'transformer':
            transformer_data = arch_data['bpt'].values
        else:
            serial_data = arch_data['bpt'].values

    data_to_plot = []
    labels = []
    if len(transformer_data) > 0:
        data_to_plot.append(transformer_data)
        labels.append('Transformer')
    if len(serial_data) > 0:
        data_to_plot.append(serial_data)
        labels.append('Serial/SSM')

    if data_to_plot:
        bp = plt.boxplot(data_to_plot, labels=labels, patch_artist=True)
        bp['boxes'][0].set_facecolor('blue') if len(data_to_plot) > 0 else None
        if len(data_to_plot) > 1:
            bp['boxes'][1].set_facecolor('red')

    plt.axhspan(3, 6, alpha=0.2, color='green', label='Basin [3,6] bits')
    plt.ylabel('BPT')
    plt.title('Experiment 3: BPT Variance Across 7 Corpora by Architecture')
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'exp3_bpt_variance.png', dpi=300)
    logger.info(f"Saved {PLOTS_DIR / 'exp3_bpt_variance.png'}")
    plt.close()

    # Plot 4: BPT vs energy scatter
    plt.figure(figsize=(10, 8))

    # Merge BPT and energy data
    df_merged = df_bpt.merge(df_energy, on=['model', 'arch_type'])

    colors = {'transformer': 'blue', 'serial': 'red'}
    for arch_type in df_merged['arch_type'].unique():
        arch_data = df_merged[df_merged['arch_type'] == arch_type]
        plt.scatter(arch_data['bpt_mean'], arch_data['energy_per_token_mJ'],
                   c=colors[arch_type], label=arch_type, s=100, alpha=0.6)

    plt.axvspan(3, 6, alpha=0.2, color='green', label='Basin [3,6] bits')
    plt.xlabel('BPT')
    plt.ylabel('Energy per Token (mJ)')
    plt.title('Experiment 3: BPT vs Energy per Token')
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'exp3_bpt_vs_energy.png', dpi=300)
    logger.info(f"Saved {PLOTS_DIR / 'exp3_bpt_vs_energy.png'}")
    plt.close()

    # Plot 5: Structural bonus (all_shuffled - original)
    plt.figure(figsize=(10, 8))

    structural_bonus = []
    for model in df_shuffling['model'].unique():
        model_data = df_shuffling[df_shuffling['model'] == model]
        original_bpt = model_data[model_data['shuffle_level'] == 'original']['bpt'].values
        shuffled_bpt = model_data[model_data['shuffle_level'] == 'all_shuffled']['bpt'].values

        if len(original_bpt) > 0 and len(shuffled_bpt) > 0:
            bonus = shuffled_bpt[0] - original_bpt[0]
            arch_type = model_data['arch_type'].iloc[0]
            structural_bonus.append({
                'model': model.split('/')[-1],
                'arch_type': arch_type,
                'bonus': bonus
            })

    df_bonus = pd.DataFrame(structural_bonus)

    # Group by architecture
    transformer_bonus = df_bonus[df_bonus['arch_type'] == 'transformer']['bonus'].values
    serial_bonus = df_bonus[df_bonus['arch_type'] == 'serial']['bonus'].values

    x = np.arange(2)
    width = 0.35

    plt.bar(x[0], np.mean(transformer_bonus) if len(transformer_bonus) > 0 else 0,
           width, label='Transformer', color='blue',
           yerr=np.std(transformer_bonus) if len(transformer_bonus) > 0 else 0)
    plt.bar(x[1], np.mean(serial_bonus) if len(serial_bonus) > 0 else 0,
           width, label='Serial/SSM', color='red',
           yerr=np.std(serial_bonus) if len(serial_bonus) > 0 else 0)

    plt.ylabel('Structural Bonus (bits)')
    plt.title('Experiment 3: Structural Bonus (Shuffled - Original)')
    plt.xticks(x, ['Transformer', 'Serial/SSM'])
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'exp3_structural_bonus.png', dpi=300)
    logger.info(f"Saved {PLOTS_DIR / 'exp3_structural_bonus.png'}")
    plt.close()


def run_statistical_analysis(df_bpt, df_shuffling, df_seven):
    """Run statistical tests and save summary"""
    logger.info("\nRunning statistical analysis...")

    stats_summary = []

    # 1. Welch's t-test: mean BPT difference between architectures
    transformer_bpt = df_bpt[df_bpt['arch_type'] == 'transformer']['bpt_mean'].values
    serial_bpt = df_bpt[df_bpt['arch_type'] == 'serial']['bpt_mean'].values

    if len(transformer_bpt) > 0 and len(serial_bpt) > 0:
        t_stat, p_value = stats.ttest_ind(transformer_bpt, serial_bpt, equal_var=False)
        stats_summary.append({
            'test': 'Welch\'s t-test (BPT)',
            'statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'interpretation': f"Transformer mean: {np.mean(transformer_bpt):.3f}, Serial mean: {np.mean(serial_bpt):.3f}"
        })

        # Cohen's d effect size
        pooled_std = np.sqrt((np.std(transformer_bpt)**2 + np.std(serial_bpt)**2) / 2)
        cohens_d = (np.mean(transformer_bpt) - np.mean(serial_bpt)) / pooled_std if pooled_std > 0 else 0
        stats_summary.append({
            'test': 'Cohen\'s d (BPT)',
            'statistic': cohens_d,
            'p_value': np.nan,
            'significant': abs(cohens_d) > 0.5,
            'interpretation': f"Effect size: {abs(cohens_d):.3f} ({'small' if abs(cohens_d) < 0.5 else 'medium' if abs(cohens_d) < 0.8 else 'large'})"
        })

    # 2. Levene's test: BPT variance difference
    transformer_corpus_bpt = df_seven[df_seven['arch_type'] == 'transformer']['bpt'].values
    serial_corpus_bpt = df_seven[df_seven['arch_type'] == 'serial']['bpt'].values

    if len(transformer_corpus_bpt) > 0 and len(serial_corpus_bpt) > 0:
        lev_stat, lev_p = stats.levene(transformer_corpus_bpt, serial_corpus_bpt)
        stats_summary.append({
            'test': 'Levene\'s test (variance)',
            'statistic': lev_stat,
            'p_value': lev_p,
            'significant': lev_p < 0.05,
            'interpretation': f"Transformer variance: {np.var(transformer_corpus_bpt):.3f}, Serial variance: {np.var(serial_corpus_bpt):.3f}"
        })

    # 3. Mann-Whitney U on structural bonus
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

    if len(transformer_bonus) > 0 and len(serial_bonus) > 0:
        u_stat, u_p = stats.mannwhitneyu(transformer_bonus, serial_bonus, alternative='two-sided')
        stats_summary.append({
            'test': 'Mann-Whitney U (structural bonus)',
            'statistic': u_stat,
            'p_value': u_p,
            'significant': u_p < 0.05,
            'interpretation': f"Transformer bonus: {np.mean(transformer_bonus):.3f}±{np.std(transformer_bonus):.3f}, Serial bonus: {np.mean(serial_bonus):.3f}±{np.std(serial_bonus):.3f}"
        })

    # Save statistical summary
    df_stats = pd.DataFrame(stats_summary)
    df_stats.to_csv(RESULTS_DIR / 'exp3_statistics.csv', index=False)
    logger.info(f"Saved {RESULTS_DIR / 'exp3_statistics.csv'}")

    return df_stats


if __name__ == '__main__':
    logger.info(f"Starting Experiment 3 at {datetime.now()}")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    run_experiment_3()

    logger.info(f"Completed at {datetime.now()}")
