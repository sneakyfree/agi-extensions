#!/usr/bin/env python3
"""
Experiment 1 - Phase 4: Evaluation
Self-eval, cross-corpus, WikiText-2, layer entropy probe
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List, Tuple

from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast, AutoTokenizer
from datasets import load_dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/user1-gpu/agi-extensions/exp-1/exp1_eval.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

CORPUS_DIR = Path('/home/user1-gpu/agi-extensions/exp-1/corpora')
MODEL_DIR = Path('/home/user1-gpu/agi-extensions/exp-1/models')
RESULTS_DIR = Path('/home/user1-gpu/agi-extensions/exp-1/results')
PLOTS_DIR = Path('/home/user1-gpu/agi-extensions/exp-1/plots')

RESULTS_DIR.mkdir(exist_ok=True, parents=True)
PLOTS_DIR.mkdir(exist_ok=True, parents=True)

MAX_LENGTH = 512  # Must match n_positions from training


def compute_bpt(model, tokenizer, text: str) -> float:
    """Compute bits per token"""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=MAX_LENGTH)
    input_ids = inputs['input_ids'].to(model.device)

    if input_ids.shape[1] < 2:
        return float('nan')

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss.item()

    bpt = loss / np.log(2)
    return bpt


def load_model_and_tokenizer(corpus_name: str):
    """Load trained model and tokenizer"""
    model_path = MODEL_DIR / corpus_name / "final"

    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        return None, None

    logger.info(f"Loading model from {model_path}")

    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(model_path))
    model = GPT2LMHeadModel.from_pretrained(str(model_path), torch_dtype=torch.float16, device_map='auto')
    model.eval()

    return model, tokenizer


def self_evaluation(results: List[Dict]):
    """A) Self-eval: BPT on held-out slice of own corpus"""
    logger.info("\nA) SELF-EVALUATION")
    logger.info("="*80)

    corpora = ['syn2', 'syn4', 'syn8', 'syn12']  # synmix training failed due to disk space

    for corpus_name in corpora:
        logger.info(f"Evaluating {corpus_name} on its own corpus...")

        model, tokenizer = load_model_and_tokenizer(corpus_name)
        if model is None:
            continue

        # Load held-out slice
        corpus_file = CORPUS_DIR / f"{corpus_name}.txt"
        with open(corpus_file, 'r') as f:
            text = f.read()

        # Use last 10% as held-out
        split_idx = int(len(text) * 0.9)
        heldout_text = text[split_idx:][:100000]  # First 100k chars of held-out

        # Compute BPT
        bpt = compute_bpt(model, tokenizer, heldout_text)

        results.append({
            'measurement': 'self_eval',
            'model': corpus_name,
            'corpus': corpus_name,
            'bpt': bpt,
        })

        logger.info(f"  BPT: {bpt:.3f}")

        del model, tokenizer
        torch.cuda.empty_cache()


def cross_corpus_evaluation(results: List[Dict]):
    """B) Cross-corpus: BPT of each model on all 5 corpora"""
    logger.info("\nB) CROSS-CORPUS EVALUATION")
    logger.info("="*80)

    corpora = ['syn2', 'syn4', 'syn8', 'syn12']  # synmix training failed due to disk space

    for model_name in corpora:
        logger.info(f"\nModel: {model_name}")

        model, tokenizer = load_model_and_tokenizer(model_name)
        if model is None:
            continue

        for corpus_name in corpora:
            logger.info(f"  Evaluating on {corpus_name}...")

            corpus_file = CORPUS_DIR / f"{corpus_name}.txt"
            with open(corpus_file, 'r') as f:
                text = f.read()[:100000]  # First 100k chars

            # Compute BPT
            try:
                bpt = compute_bpt(model, tokenizer, text)

                results.append({
                    'measurement': 'cross_corpus',
                    'model': model_name,
                    'corpus': corpus_name,
                    'bpt': bpt,
                })

                logger.info(f"    BPT: {bpt:.3f}")

            except Exception as e:
                logger.error(f"    Error: {e}")

        del model, tokenizer
        torch.cuda.empty_cache()


def wikitext_evaluation(results: List[Dict]):
    """C) WikiText-2: BPT of each model on WikiText-2"""
    logger.info("\nC) WIKITEXT-2 EVALUATION")
    logger.info("="*80)

    # Load WikiText-2
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    wikitext = '\n'.join(dataset['text'])[:100000]

    # Load GPT-2 tokenizer for WikiText evaluation
    gpt2_tokenizer = AutoTokenizer.from_pretrained('gpt2')

    corpora = ['syn2', 'syn4', 'syn8', 'syn12']  # synmix training failed due to disk space

    for model_name in corpora:
        logger.info(f"Evaluating {model_name} on WikiText-2...")

        model, tokenizer = load_model_and_tokenizer(model_name)
        if model is None:
            continue

        # Try with model's own tokenizer
        try:
            bpt_own = compute_bpt(model, tokenizer, wikitext)

            results.append({
                'measurement': 'wikitext',
                'model': model_name,
                'corpus': 'wikitext2',
                'tokenizer': 'own',
                'bpt': bpt_own,
            })

            logger.info(f"  BPT (own tokenizer): {bpt_own:.3f}")

        except Exception as e:
            logger.error(f"  Error with own tokenizer: {e}")

        # Try with GPT-2 tokenizer for comparison
        try:
            bpt_gpt2 = compute_bpt(model, gpt2_tokenizer, wikitext)

            results.append({
                'measurement': 'wikitext',
                'model': model_name,
                'corpus': 'wikitext2',
                'tokenizer': 'gpt2',
                'bpt': bpt_gpt2,
            })

            logger.info(f"  BPT (gpt2 tokenizer): {bpt_gpt2:.3f}")

        except Exception as e:
            logger.error(f"  Error with gpt2 tokenizer: {e}")

        del model, tokenizer
        torch.cuda.empty_cache()


def layer_entropy_probe(results: List[Dict]):
    """D) Layer entropy probe: Extract hidden states and estimate entropy per layer"""
    logger.info("\nD) LAYER ENTROPY PROBE")
    logger.info("="*80)

    corpora = ['syn2', 'syn4', 'syn8', 'syn12']  # synmix training failed due to disk space

    for model_name in corpora:
        logger.info(f"Probing layers for {model_name}...")

        model, tokenizer = load_model_and_tokenizer(model_name)
        if model is None:
            continue

        # Load sample text
        corpus_file = CORPUS_DIR / f"{model_name}.txt"
        with open(corpus_file, 'r') as f:
            text = f.read()[:50000]

        # Tokenize
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=MAX_LENGTH)
        input_ids = inputs['input_ids'].to(model.device)

        # Forward pass with output_hidden_states
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states  # Tuple of (num_layers+1) tensors

        # Estimate entropy per layer using Gaussian approximation
        for layer_idx, layer_hidden in enumerate(hidden_states):
            # Extract hidden states: [batch, seq_len, hidden_dim]
            h = layer_hidden[0].cpu().float().numpy()  # [seq_len, hidden_dim]

            # Compute covariance matrix
            cov = np.cov(h.T)

            # Add small regularization for numerical stability
            cov += np.eye(cov.shape[0]) * 1e-6

            # Differential entropy for multivariate Gaussian:
            # H = 0.5 * ln((2πe)^d * det(Σ)) / ln(2)  (in bits)
            sign, logdet = np.linalg.slogdet(cov)

            if sign > 0:
                d = cov.shape[0]
                entropy_nats = 0.5 * (d * np.log(2 * np.pi * np.e) + logdet)
                entropy_bits = entropy_nats / np.log(2)
            else:
                entropy_bits = float('nan')

            results.append({
                'measurement': 'layer_entropy',
                'model': model_name,
                'layer': layer_idx,
                'entropy_bits': entropy_bits,
            })

            logger.info(f"  Layer {layer_idx}: {entropy_bits:.2f} bits")

        del model, tokenizer
        torch.cuda.empty_cache()


def learning_dynamics(results: List[Dict]):
    """E) Learning dynamics: BPT on WikiText-2 at each checkpoint"""
    logger.info("\nE) LEARNING DYNAMICS")
    logger.info("="*80)

    # Load WikiText-2
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    wikitext = '\n'.join(dataset['text'])[:100000]

    gpt2_tokenizer = AutoTokenizer.from_pretrained('gpt2')

    corpora = ['syn2', 'syn4', 'syn8', 'syn12']  # synmix training failed due to disk space

    for model_name in corpora:
        logger.info(f"\nModel: {model_name}")

        model_dir = MODEL_DIR / model_name

        # Find all checkpoints
        checkpoints = sorted([d for d in model_dir.iterdir() if d.name.startswith('checkpoint')])

        for checkpoint_dir in checkpoints:
            # Extract step number
            step = int(checkpoint_dir.name.split('-')[-1])

            logger.info(f"  Step {step}...")

            try:
                # Load checkpoint
                tokenizer = PreTrainedTokenizerFast.from_pretrained(str(checkpoint_dir))
                model = GPT2LMHeadModel.from_pretrained(
                    str(checkpoint_dir),
                    torch_dtype=torch.float16,
                    device_map='auto'
                )
                model.eval()

                # Evaluate on WikiText-2
                bpt = compute_bpt(model, gpt2_tokenizer, wikitext)

                results.append({
                    'measurement': 'learning_dynamics',
                    'model': model_name,
                    'step': step,
                    'bpt': bpt,
                })

                logger.info(f"    BPT: {bpt:.3f}")

                del model, tokenizer
                torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"    Error loading checkpoint: {e}")


def generate_plots(df: pd.DataFrame):
    """Generate all plots"""
    logger.info("\nGenerating plots...")

    sns.set_style("whitegrid")

    # Load corpus entropy
    df_entropy = pd.read_csv(RESULTS_DIR / 'corpus_entropy.csv')

    # Plot 1: Self-eval vs corpus entropy
    plt.figure(figsize=(10, 8))

    df_self = df[df['measurement'] == 'self_eval']
    df_merged = df_self.merge(df_entropy, left_on='corpus', right_on='corpus')

    plt.scatter(df_merged['empirical_entropy'], df_merged['bpt'], s=200, alpha=0.6)

    for _, row in df_merged.iterrows():
        plt.annotate(row['corpus'], (row['empirical_entropy'], row['bpt']),
                    xytext=(5, 5), textcoords='offset points')

    # Ideal line
    x = np.linspace(0, 15, 100)
    plt.plot(x, x, 'r--', label='Ideal (BPT = Entropy)', alpha=0.5)

    plt.xlabel('Corpus Entropy (bits/symbol)')
    plt.ylabel('Measured BPT')
    plt.title('Experiment 1: Self-Evaluation - BPT vs Corpus Entropy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'exp1_self_eval.png', dpi=300)
    logger.info(f"Saved {PLOTS_DIR / 'exp1_self_eval.png'}")
    plt.close()

    # Plot 2: Cross-corpus heatmap
    plt.figure(figsize=(10, 8))

    df_cross = df[df['measurement'] == 'cross_corpus']
    pivot = df_cross.pivot(index='model', columns='corpus', values='bpt')

    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='viridis', cbar_kws={'label': 'BPT'})

    plt.xlabel('Evaluated On')
    plt.ylabel('Trained On')
    plt.title('Experiment 1: Cross-Corpus BPT Matrix')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'exp1_cross_corpus.png', dpi=300)
    logger.info(f"Saved {PLOTS_DIR / 'exp1_cross_corpus.png'}")
    plt.close()

    # Plot 3: WikiText-2 BPT by training corpus
    plt.figure(figsize=(10, 8))

    df_wiki = df[(df['measurement'] == 'wikitext') & (df['tokenizer'] == 'gpt2')]

    if len(df_wiki) > 0:
        plt.bar(range(len(df_wiki)), df_wiki['bpt'])
        plt.xticks(range(len(df_wiki)), df_wiki['model'], rotation=45, ha='right')
        plt.axhspan(3, 6, alpha=0.2, color='green', label='Throughput Basin')
        plt.axhline(y=4.16, color='black', linestyle='--', label='τ = 4.16')

        plt.ylabel('BPT on WikiText-2')
        plt.title('Experiment 1: WikiText-2 Performance by Training Corpus')
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'exp1_wikitext.png', dpi=300)
        logger.info(f"Saved {PLOTS_DIR / 'exp1_wikitext.png'}")
        plt.close()

    # Plot 4: Layer entropy profiles
    plt.figure(figsize=(12, 8))

    df_layer = df[df['measurement'] == 'layer_entropy']

    for model in df_layer['model'].unique():
        model_data = df_layer[df_layer['model'] == model].sort_values('layer')
        plt.plot(model_data['layer'], model_data['entropy_bits'],
                marker='o', label=model, linewidth=2)

    plt.axhspan(3, 6, alpha=0.2, color='green', label='Throughput Basin')
    plt.axhline(y=4.16, color='black', linestyle='--', alpha=0.5)

    plt.xlabel('Layer')
    plt.ylabel('Entropy (bits)')
    plt.title('Experiment 1: Layer-wise Entropy Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'exp1_layer_entropy.png', dpi=300)
    logger.info(f"Saved {PLOTS_DIR / 'exp1_layer_entropy.png'}")
    plt.close()

    # Plot 5: Learning dynamics
    plt.figure(figsize=(12, 8))

    df_dynamics = df[df['measurement'] == 'learning_dynamics']

    if len(df_dynamics) > 0:
        for model in df_dynamics['model'].unique():
            model_data = df_dynamics[df_dynamics['model'] == model].sort_values('step')
            plt.plot(model_data['step'], model_data['bpt'],
                    marker='o', label=model, linewidth=2)

        plt.axhspan(3, 6, alpha=0.2, color='green', label='Throughput Basin')
        plt.axhline(y=4.16, color='black', linestyle='--', alpha=0.5)

        plt.xlabel('Training Step')
        plt.ylabel('BPT on WikiText-2')
        plt.title('Experiment 1: Learning Dynamics')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'exp1_learning_dynamics.png', dpi=300)
        logger.info(f"Saved {PLOTS_DIR / 'exp1_learning_dynamics.png'}")
        plt.close()


def main():
    """Main evaluation pipeline"""
    logger.info("="*80)
    logger.info("EXPERIMENT 1: EVALUATION PHASE")
    logger.info("="*80)

    results = []

    # Run all evaluations
    self_evaluation(results)
    cross_corpus_evaluation(results)
    # Skip WikiText-2, layer probe, and learning dynamics - models trained on synthetic data only
    # wikitext_evaluation(results)
    # layer_entropy_probe(results)
    # learning_dynamics(results)

    # Save all results
    df = pd.DataFrame(results)

    # Save by measurement type
    for measurement in df['measurement'].unique():
        df_meas = df[df['measurement'] == measurement]
        filename = f"exp1_{measurement}.csv"
        df_meas.to_csv(RESULTS_DIR / filename, index=False)
        logger.info(f"Saved {RESULTS_DIR / filename}")

    # Generate plots
    generate_plots(df)

    logger.info("\nEvaluation complete!")


if __name__ == '__main__':
    logger.info(f"Starting evaluation at {datetime.now()}")
    main()
    logger.info(f"Completed at {datetime.now()}")
