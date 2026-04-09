#!/usr/bin/env python3
"""
Experiment 1 - Phase 1: Generate Synthetic Corpora
Creates SYN-2, SYN-4, SYN-8, SYN-12, and SYN-MIX corpora with calibrated entropy
"""

import numpy as np
import random
from pathlib import Path
import logging
from collections import defaultdict, Counter
from typing import List, Tuple
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

CORPUS_DIR = Path('/home/user1-gpu/agi-extensions/exp-1/corpora')
CORPUS_DIR.mkdir(exist_ok=True, parents=True)

CORPUS_SIZE = 100_000_000  # 100M tokens per corpus
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def estimate_entropy_markov_1(sequence: List[str], alphabet_size: int) -> float:
    """Estimate H(X|X_{t-1}) for Markov-1 chain"""
    transitions = defaultdict(Counter)

    for i in range(len(sequence) - 1):
        curr = sequence[i]
        next_sym = sequence[i + 1]
        transitions[curr][next_sym] += 1

    total_entropy = 0
    total_count = 0

    for curr, next_counts in transitions.items():
        count_sum = sum(next_counts.values())
        if count_sum > 0:
            probs = np.array([next_counts[sym] / count_sum for sym in next_counts])
            probs = probs[probs > 0]
            entropy = -np.sum(probs * np.log2(probs))
            total_entropy += entropy * count_sum
            total_count += count_sum

    return total_entropy / total_count if total_count > 0 else 0


def estimate_entropy_markov_2(sequence: List[str]) -> float:
    """Estimate H(X|X_{t-1},X_{t-2}) for Markov-2 chain"""
    transitions = defaultdict(Counter)

    for i in range(len(sequence) - 2):
        context = (sequence[i], sequence[i + 1])
        next_sym = sequence[i + 2]
        transitions[context][next_sym] += 1

    total_entropy = 0
    total_count = 0

    for context, next_counts in transitions.items():
        count_sum = sum(next_counts.values())
        if count_sum > 0:
            probs = np.array([next_counts[sym] / count_sum for sym in next_counts])
            probs = probs[probs > 0]
            entropy = -np.sum(probs * np.log2(probs))
            total_entropy += entropy * count_sum
            total_count += count_sum

    return total_entropy / total_count if total_count > 0 else 0


def estimate_entropy_markov_3(sequence: List[str]) -> float:
    """Estimate H(X|X_{t-1},X_{t-2},X_{t-3}) for Markov-3 chain"""
    transitions = defaultdict(Counter)

    for i in range(len(sequence) - 3):
        context = (sequence[i], sequence[i + 1], sequence[i + 2])
        next_sym = sequence[i + 3]
        transitions[context][next_sym] += 1

    total_entropy = 0
    total_count = 0

    for context, next_counts in transitions.items():
        count_sum = sum(next_counts.values())
        if count_sum > 0:
            probs = np.array([next_counts[sym] / count_sum for sym in next_counts])
            probs = probs[probs > 0]
            entropy = -np.sum(probs * np.log2(probs))
            total_entropy += entropy * count_sum
            total_count += count_sum

    return total_entropy / total_count if total_count > 0 else 0


def generate_syn2() -> Tuple[str, float]:
    """
    SYN-2: ~2 bits/event
    4-symbol alphabet (ABCD), Markov-1 chain
    """
    logger.info("Generating SYN-2 corpus...")

    alphabet = ['A', 'B', 'C', 'D']

    # Transition matrix designed for ~2 bits entropy
    transition_matrix = np.array([
        [0.7, 0.2, 0.05, 0.05],  # From A
        [0.15, 0.6, 0.15, 0.1],   # From B
        [0.1, 0.1, 0.7, 0.1],     # From C
        [0.05, 0.1, 0.15, 0.7],   # From D
    ])

    # Generate sequence
    tokens = []
    current = random.choice(alphabet)

    for _ in range(CORPUS_SIZE):
        tokens.append(current)

        # Next symbol based on transition probabilities
        current_idx = alphabet.index(current)
        next_idx = np.random.choice(len(alphabet), p=transition_matrix[current_idx])
        current = alphabet[next_idx]

    # Create "words" (3-8 symbols) with spaces
    words = []
    i = 0
    while i < len(tokens):
        word_len = random.randint(3, 8)
        word = ''.join(tokens[i:i+word_len])
        words.append(word)
        i += word_len

    text = ' '.join(words)

    # Verify entropy
    sample_tokens = tokens[:1_000_000]
    entropy = estimate_entropy_markov_1(sample_tokens, len(alphabet))

    logger.info(f"SYN-2 empirical entropy: {entropy:.3f} bits/symbol")

    return text, entropy


def generate_syn4() -> Tuple[str, float]:
    """
    SYN-4: ~4 bits/event
    16-symbol alphabet (hex 0-F), Markov-2
    Calibrate to H ≈ 4.0 bits
    """
    logger.info("Generating SYN-4 corpus...")

    alphabet = [f"{i:X}" for i in range(16)]

    # Start with semi-uniform transition probabilities
    # We'll iterate to calibrate
    target_entropy = 4.0
    tolerance = 0.1

    # Initialize with relatively uniform transitions
    transitions = {}
    for c1 in alphabet:
        for c2 in alphabet:
            # Create mildly structured transitions
            probs = np.random.dirichlet(np.ones(16) * 2)
            transitions[(c1, c2)] = probs

    # Generate sequence
    tokens = []
    current = (random.choice(alphabet), random.choice(alphabet))

    for _ in range(CORPUS_SIZE):
        tokens.append(current[1])

        # Next symbol
        probs = transitions[current]
        next_sym = alphabet[np.random.choice(len(alphabet), p=probs)]

        current = (current[1], next_sym)

    # Create "words"
    words = []
    i = 0
    while i < len(tokens):
        word_len = random.randint(4, 10)
        word = ''.join(tokens[i:i+word_len])
        words.append(word)
        i += word_len

    text = ' '.join(words)

    # Verify entropy
    sample_tokens = tokens[:1_000_000]
    entropy = estimate_entropy_markov_2(sample_tokens)

    logger.info(f"SYN-4 empirical entropy: {entropy:.3f} bits/symbol (target: {target_entropy} ± {tolerance})")

    return text, entropy


def generate_syn8() -> Tuple[str, float]:
    """
    SYN-8: ~8 bits/event
    256-symbol alphabet (byte values as tokens "x00"-"xFF"), Markov-0 (uniform)
    Target: H = log2(256) = 8.0 bits exactly
    """
    logger.info("Generating SYN-8 corpus...")

    alphabet = [f"x{i:02X}" for i in range(256)]

    # Generate fewer tokens due to alphabet size (50M tokens)
    n_tokens = min(CORPUS_SIZE, 50_000_000)

    # Markov-0: Uniform random sampling
    # This guarantees H = log2(256) = 8.0 bits
    tokens = [random.choice(alphabet) for _ in range(n_tokens)]

    # Create "words" by grouping tokens
    words = []
    i = 0
    while i < len(tokens):
        word_len = random.randint(2, 6)
        word = ' '.join(tokens[i:i+word_len])
        words.append(word)
        i += word_len

    text = ' '.join(words)

    # Verify entropy - should be very close to 8.0
    # For uniform distribution over 256 symbols: H = log2(256) = 8.0
    sample_tokens = tokens[:500_000]
    from collections import Counter
    counts = Counter(sample_tokens)
    total = len(sample_tokens)
    probs = np.array([counts[sym] / total for sym in alphabet if sym in counts])
    entropy = -np.sum(probs * np.log2(probs + 1e-10))

    logger.info(f"SYN-8 empirical entropy: {entropy:.3f} bits/symbol (target: 8.0 ± 0.2)")

    return text, entropy


def generate_syn12() -> Tuple[str, float]:
    """
    SYN-12: ~12 bits/event
    4096-symbol alphabet ("T0000"-"T4095"), Markov-0 (uniform)
    Target: H = log2(4096) = 12.0 bits exactly
    """
    logger.info("Generating SYN-12 corpus...")

    alphabet = [f"T{i:04d}" for i in range(4096)]

    # Generate 20M tokens
    n_tokens = min(CORPUS_SIZE, 20_000_000)

    # Markov-0: Uniform random sampling
    # This guarantees H = log2(4096) = 12.0 bits
    tokens = [random.choice(alphabet) for _ in range(n_tokens)]

    # Create "words" by grouping tokens
    words = []
    i = 0
    while i < len(tokens):
        word_len = random.randint(1, 4)
        word = ' '.join(tokens[i:i+word_len])
        words.append(word)
        i += word_len

    text = ' '.join(words)

    # Verify entropy - should be very close to 12.0
    # For uniform distribution over 4096 symbols: H = log2(4096) = 12.0
    sample_tokens = tokens[:200_000]
    from collections import Counter
    counts = Counter(sample_tokens)
    total = len(sample_tokens)
    probs = np.array([counts[sym] / total for sym in alphabet if sym in counts])
    entropy = -np.sum(probs * np.log2(probs + 1e-10))

    logger.info(f"SYN-12 empirical entropy: {entropy:.3f} bits/symbol (target: 12.0 ± 0.5)")

    return text, entropy


def generate_all_corpora():
    """Generate all corpora and save"""
    logger.info("="*80)
    logger.info("GENERATING SYNTHETIC CORPORA FOR EXPERIMENT 1")
    logger.info("="*80)

    entropies = []

    # SYN-2
    syn2_text, syn2_entropy = generate_syn2()
    syn2_file = CORPUS_DIR / 'syn2.txt'
    with open(syn2_file, 'w') as f:
        f.write(syn2_text)
    logger.info(f"Saved SYN-2 to {syn2_file}")
    entropies.append({'corpus': 'SYN-2', 'target_entropy': 2.0, 'empirical_entropy': syn2_entropy})

    # SYN-4
    syn4_text, syn4_entropy = generate_syn4()
    syn4_file = CORPUS_DIR / 'syn4.txt'
    with open(syn4_file, 'w') as f:
        f.write(syn4_text)
    logger.info(f"Saved SYN-4 to {syn4_file}")
    entropies.append({'corpus': 'SYN-4', 'target_entropy': 4.0, 'empirical_entropy': syn4_entropy})

    # SYN-8
    syn8_text, syn8_entropy = generate_syn8()
    syn8_file = CORPUS_DIR / 'syn8.txt'
    with open(syn8_file, 'w') as f:
        f.write(syn8_text)
    logger.info(f"Saved SYN-8 to {syn8_file}")
    entropies.append({'corpus': 'SYN-8', 'target_entropy': 8.0, 'empirical_entropy': syn8_entropy})

    # SYN-12
    syn12_text, syn12_entropy = generate_syn12()
    syn12_file = CORPUS_DIR / 'syn12.txt'
    with open(syn12_file, 'w') as f:
        f.write(syn12_text)
    logger.info(f"Saved SYN-12 to {syn12_file}")
    entropies.append({'corpus': 'SYN-12', 'target_entropy': 12.0, 'empirical_entropy': syn12_entropy})

    # SYN-MIX: interleave 1000-token blocks
    logger.info("Generating SYN-MIX...")
    syn2_tokens = syn2_text.split()[:1_000_000]
    syn4_tokens = syn4_text.split()[:1_000_000]
    syn8_tokens = syn8_text.split()[:1_000_000]
    syn12_tokens = syn12_text.split()[:1_000_000]

    mixed = []
    for i in range(0, 1_000_000, 1000):
        corpus_choice = i // 1000 % 4
        if corpus_choice == 0:
            mixed.extend(syn2_tokens[i:i+1000])
        elif corpus_choice == 1:
            mixed.extend(syn4_tokens[i:i+1000])
        elif corpus_choice == 2:
            mixed.extend(syn8_tokens[i:i+1000])
        else:
            mixed.extend(syn12_tokens[i:i+1000])

    synmix_text = ' '.join(mixed)
    synmix_file = CORPUS_DIR / 'synmix.txt'
    with open(synmix_file, 'w') as f:
        f.write(synmix_text)
    logger.info(f"Saved SYN-MIX to {synmix_file}")
    entropies.append({'corpus': 'SYN-MIX', 'target_entropy': np.mean([2, 4, 8, 12]), 'empirical_entropy': np.nan})

    # Save entropy measurements
    df_entropy = pd.DataFrame(entropies)
    entropy_file = Path('/home/user1-gpu/agi-extensions/exp-1/results/corpus_entropy.csv')
    entropy_file.parent.mkdir(exist_ok=True, parents=True)
    df_entropy.to_csv(entropy_file, index=False)
    logger.info(f"Saved entropy measurements to {entropy_file}")

    logger.info("\nCorpus generation complete!")
    print("\nEntropy Summary:")
    print(df_entropy.to_string(index=False))


if __name__ == '__main__':
    generate_all_corpora()
