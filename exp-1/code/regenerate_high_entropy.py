#!/usr/bin/env python3
"""
Regenerate only SYN-8, SYN-12, and SYN-MIX with fixed entropy
"""
import sys
from pathlib import Path
import pandas as pd

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from exp1_generate_corpora import (
    generate_syn8, generate_syn12, CORPUS_DIR, logger
)
import random

def regenerate():
    """Regenerate high-entropy corpora"""
    
    # SYN-8
    logger.info("Regenerating SYN-8...")
    syn8_text, syn8_entropy = generate_syn8()
    syn8_file = CORPUS_DIR / 'syn8.txt'
    with open(syn8_file, 'w') as f:
        f.write(syn8_text)
    logger.info(f"Saved SYN-8 to {syn8_file}")
    
    # SYN-12
    logger.info("Regenerating SYN-12...")
    syn12_text, syn12_entropy = generate_syn12()
    syn12_file = CORPUS_DIR / 'syn12.txt'
    with open(syn12_file, 'w') as f:
        f.write(syn12_text)
    logger.info(f"Saved SYN-12 to {syn12_file}")
    
    # SYN-MIX
    logger.info("Regenerating SYN-MIX...")
    syn2_file = CORPUS_DIR / 'syn2.txt'
    syn4_file = CORPUS_DIR / 'syn4.txt'
    
    with open(syn2_file) as f:
        syn2 = f.read().split()[:100_000]
    with open(syn4_file) as f:
        syn4 = f.read().split()[:100_000]
    
    syn8_sample = syn8_text.split()[:100_000]
    syn12_sample = syn12_text.split()[:100_000]
    
    # Interleave
    synmix_words = []
    for i in range(max(len(syn2), len(syn4), len(syn8_sample), len(syn12_sample))):
        if i < len(syn2):
            synmix_words.append(syn2[i])
        if i < len(syn4):
            synmix_words.append(syn4[i])
        if i < len(syn8_sample):
            synmix_words.append(syn8_sample[i])
        if i < len(syn12_sample):
            synmix_words.append(syn12_sample[i])
    
    synmix_text = ' '.join(synmix_words)
    synmix_file = CORPUS_DIR / 'synmix.txt'
    with open(synmix_file, 'w') as f:
        f.write(synmix_text)
    logger.info(f"Saved SYN-MIX to {synmix_file}")
    
    # Update entropy CSV
    df = pd.read_csv(CORPUS_DIR.parent / 'results' / 'corpus_entropy.csv')
    df.loc[df['corpus'] == 'SYN-8', 'empirical_entropy'] = syn8_entropy
    df.loc[df['corpus'] == 'SYN-12', 'empirical_entropy'] = syn12_entropy
    df.to_csv(CORPUS_DIR.parent / 'results' / 'corpus_entropy.csv', index=False)
    
    logger.info("\n" + "="*80)
    logger.info("Regeneration complete!")
    logger.info("="*80)

if __name__ == "__main__":
    regenerate()
