#!/usr/bin/env python3
"""
Experiment 1 - Phases 2-3: Train Tokenizers and Models
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/user1-gpu/agi-extensions/exp-1/exp1_train.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

CORPUS_DIR = Path('/home/user1-gpu/agi-extensions/exp-1/corpora')
TOKENIZER_DIR = Path('/home/user1-gpu/agi-extensions/exp-1/tokenizers')
MODEL_DIR = Path('/home/user1-gpu/agi-extensions/exp-1/models')

TOKENIZER_DIR.mkdir(exist_ok=True, parents=True)
MODEL_DIR.mkdir(exist_ok=True, parents=True)

VOCAB_SIZE = 8192
SEED = 42

# Training hyperparameters
N_EMBD = 768
N_LAYER = 12
N_HEAD = 12
BATCH_SIZE = 32
SEQ_LEN = 512
LEARNING_RATE = 3e-4
WARMUP_STEPS = 1000
MAX_STEPS = 50000
SAVE_STEPS = 5000


def train_tokenizer(corpus_name: str, corpus_file: Path) -> PreTrainedTokenizerFast:
    """Train BPE tokenizer on corpus"""
    logger.info(f"Training tokenizer for {corpus_name}...")

    # Initialize BPE tokenizer
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    # Train
    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        show_progress=True
    )

    tokenizer.train(files=[str(corpus_file)], trainer=trainer)

    # Save
    tokenizer_path = TOKENIZER_DIR / corpus_name
    tokenizer_path.mkdir(exist_ok=True, parents=True)
    tokenizer.save(str(tokenizer_path / "tokenizer.json"))

    # Wrap in HuggingFace tokenizer
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_path / "tokenizer.json"),
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )

    hf_tokenizer.save_pretrained(str(tokenizer_path))

    logger.info(f"Saved tokenizer to {tokenizer_path}")

    return hf_tokenizer


def load_corpus_dataset(corpus_file: Path, tokenizer, test_split: float = 0.1):
    """Load corpus as dataset"""
    logger.info(f"Loading corpus from {corpus_file}...")

    with open(corpus_file, 'r') as f:
        text = f.read()

    # Split into train/test
    split_idx = int(len(text) * (1 - test_split))
    train_text = text[:split_idx]
    test_text = text[split_idx:]

    # Split text into chunks of ~10000 characters each
    # This creates multiple training examples instead of a single one
    def chunk_text(text, chunk_size=10000):
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i+chunk_size]
            if len(chunk) > 100:  # Only keep substantial chunks
                chunks.append(chunk)
        return chunks

    train_chunks = chunk_text(train_text)
    test_chunks = chunk_text(test_text)

    logger.info(f"Split into {len(train_chunks)} train chunks and {len(test_chunks)} test chunks")

    # Create datasets with multiple examples
    train_dataset = Dataset.from_dict({"text": train_chunks})
    test_dataset = Dataset.from_dict({"text": test_chunks})

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=SEQ_LEN)

    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    return train_dataset, test_dataset


def train_model(corpus_name: str, tokenizer, train_dataset, test_dataset):
    """Train GPT-2 model on corpus"""
    logger.info(f"Training model for {corpus_name}...")

    # Initialize model
    config = GPT2Config(
        vocab_size=VOCAB_SIZE,
        n_embd=N_EMBD,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_positions=SEQ_LEN,
        bos_token_id=tokenizer.cls_token_id,
        eos_token_id=tokenizer.sep_token_id,
    )

    model = GPT2LMHeadModel(config)

    logger.info(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")

    # Training arguments
    output_dir = MODEL_DIR / corpus_name
    output_dir.mkdir(exist_ok=True, parents=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=1,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=WARMUP_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        logging_dir=str(output_dir / "logs"),
        logging_steps=500,
        save_steps=SAVE_STEPS,
        save_total_limit=5,
        eval_steps=SAVE_STEPS,
        evaluation_strategy="steps",
        fp16=torch.cuda.is_available(),
        seed=SEED,
        report_to=[],  # Disable wandb/tensorboard
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal language modeling
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save final model
    final_model_path = output_dir / "final"
    trainer.save_model(str(final_model_path))
    tokenizer.save_pretrained(str(final_model_path))

    logger.info(f"Saved final model to {final_model_path}")

    return model, trainer


def main():
    """Main training pipeline"""
    logger.info("="*80)
    logger.info("EXPERIMENT 1: TRAINING PHASE")
    logger.info("="*80)

    corpora = [
        ('syn2', CORPUS_DIR / 'syn2.txt'),
        ('syn4', CORPUS_DIR / 'syn4.txt'),
        ('syn8', CORPUS_DIR / 'syn8.txt'),
        ('syn12', CORPUS_DIR / 'syn12.txt'),
        ('synmix', CORPUS_DIR / 'synmix.txt'),
    ]

    for corpus_name, corpus_file in corpora:
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing {corpus_name}")
        logger.info(f"{'='*80}")

        if not corpus_file.exists():
            logger.error(f"Corpus file {corpus_file} not found! Run exp1_generate_corpora.py first.")
            continue

        # Phase 2: Train tokenizer
        tokenizer = train_tokenizer(corpus_name, corpus_file)

        # Load dataset
        train_dataset, test_dataset = load_corpus_dataset(corpus_file, tokenizer)

        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Test dataset size: {len(test_dataset)}")

        # Phase 3: Train model
        model, trainer = train_model(corpus_name, tokenizer, train_dataset, test_dataset)

        # Cleanup
        del model
        del trainer
        torch.cuda.empty_cache()

        logger.info(f"Completed training for {corpus_name}")

    logger.info("\nAll training complete!")


if __name__ == '__main__':
    logger.info(f"Starting training at {datetime.now()}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    main()

    logger.info(f"Completed at {datetime.now()}")
