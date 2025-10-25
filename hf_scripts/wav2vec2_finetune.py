#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wav2Vec2 Finetuning Script for Bengali ASR
Created for Hugging Face Transformers

This script provides a complete pipeline for finetuning wav2vec2 models
for Bengali Automatic Speech Recognition using character-level tokenization.
"""

import os
import json
import logging
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torchaudio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm
import librosa

from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset as HFDataset, load_metric

# Suppress warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune."""
    model_name_or_path: str = field(
        default="facebook/wav2vec2-base",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained tokenizer or tokenizer identifier from huggingface.co/models"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"}
    )
    freeze_feature_extractor: bool = field(
        default=True,
        metadata={"help": "Whether to freeze the feature extractor layers of the model"}
    )
    attention_dropout: float = field(
        default=0.1,
        metadata={"help": "The dropout ratio for the attention probabilities"}
    )
    activation_dropout: float = field(
        default=0.1,
        metadata={"help": "The dropout ratio for activations inside the fully connected layer"}
    )
    hidden_dropout: float = field(
        default=0.1,
        metadata={"help": "The dropout ratio for all fully connected layers in the embeddings, encoder, and pooler"}
    )
    feat_proj_dropout: float = field(
        default=0.0,
        metadata={"help": "The dropout ratio for the projected features"}
    )
    layerdrop: float = field(
        default=0.1,
        metadata={"help": "The LayerDrop probability"}
    )
    ctc_loss_reduction: str = field(
        default="mean",
        metadata={"help": "The way the ctc loss should be reduced. Should be one of 'mean' or 'sum'"}
    )

@dataclass
class DataTrainingArguments:
    """Arguments pertaining to what data we are going to input our model for training and eval."""
    train_data_path: str = field(
        default="./files/train.tsv",
        metadata={"help": "Path to training data TSV file"}
    )
    valid_data_path: str = field(
        default="./files/valid.tsv",
        metadata={"help": "Path to validation data TSV file"}
    )
    train_labels_path: str = field(
        default="./train.ltr",
        metadata={"help": "Path to training labels file"}
    )
    valid_labels_path: str = field(
        default="./train.ltr",  # Assuming same format for validation
        metadata={"help": "Path to validation labels file"}
    )
    audio_column: str = field(
        default="path",
        metadata={"help": "The name of the dataset column containing the audio data"}
    )
    text_column: str = field(
        default="sentence",
        metadata={"help": "The name of the dataset column containing the text data"}
    )
    max_duration_in_seconds: float = field(
        default=20.0,
        metadata={"help": "Filter audio files that are longer than `max_duration_in_seconds` seconds"}
    )
    min_duration_in_seconds: float = field(
        default=1.0,
        metadata={"help": "Filter audio files that are shorter than `min_duration_in_seconds` seconds"}
    )
    preprocessing_num_workers: int = field(
        default=4,
        metadata={"help": "The number of processes to use for the preprocessing"}
    )
    chars_to_ignore: List[str] = field(
        default_factory=lambda: [",", "?", ".", "!", "-", ";", ":", '""', "'", " "],
        metadata={"help": "A list of characters to remove from the transcripts"}
    )

class BengaliASRDataset(Dataset):
    """Custom Dataset for Bengali ASR with character-level tokenization."""
    
    def __init__(
        self,
        audio_paths: List[str],
        labels: List[str],
        processor: Wav2Vec2Processor,
        sampling_rate: int = 16000,
        max_duration: float = 20.0,
        min_duration: float = 1.0,
        chars_to_ignore: List[str] = None
    ):
        self.audio_paths = audio_paths
        self.labels = labels
        self.processor = processor
        self.sampling_rate = sampling_rate
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.chars_to_ignore = chars_to_ignore or []
        
        # Filter out samples that are too short or too long
        self._filter_samples()
    
    def _filter_samples(self):
        """Filter samples based on duration constraints."""
        filtered_audio_paths = []
        filtered_labels = []
        
        for audio_path, label in zip(self.audio_paths, self.labels):
            try:
                # Get audio duration
                info = torchaudio.info(audio_path)
                duration = info.num_frames / info.sample_rate
                
                if self.min_duration <= duration <= self.max_duration:
                    filtered_audio_paths.append(audio_path)
                    filtered_labels.append(label)
            except Exception as e:
                logger.warning(f"Could not load {audio_path}: {e}")
                continue
        
        self.audio_paths = filtered_audio_paths
        self.labels = filtered_labels
        logger.info(f"Filtered dataset: {len(self.audio_paths)} samples remaining")
    
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        label = self.labels[idx]
        
        # Load audio
        try:
            audio_array, sampling_rate = torchaudio.load(audio_path)
            
            # Resample if necessary
            if sampling_rate != self.sampling_rate:
                resampler = torchaudio.transforms.Resample(sampling_rate, self.sampling_rate)
                audio_array = resampler(audio_array)
            
            # Convert to mono if stereo
            if audio_array.shape[0] > 1:
                audio_array = torch.mean(audio_array, dim=0, keepdim=True)
            
            # Convert to numpy and squeeze
            audio_array = audio_array.squeeze().numpy()
            
        except Exception as e:
            logger.warning(f"Error loading {audio_path}: {e}")
            # Return silence if loading fails
            audio_array = np.zeros(int(self.sampling_rate * 1.0))
        
        # Clean label
        for char in self.chars_to_ignore:
            label = label.replace(char, "")
        
        # Process audio and text
        processed = self.processor(
            audio=audio_array,
            sampling_rate=self.sampling_rate,
            text=label,
            padding=True,
            return_tensors="pt"
        )
        
        return {
            "input_values": processed.input_values.squeeze(),
            "attention_mask": processed.attention_mask.squeeze(),
            "labels": processed.labels.squeeze()
        }

def create_vocab_from_labels(labels: List[str], chars_to_ignore: List[str] = None) -> Dict[str, int]:
    """Create vocabulary from labels for character-level tokenization."""
    chars_to_ignore = chars_to_ignore or []
    
    # Collect all unique characters
    vocab = set()
    for label in labels:
        for char in label:
            if char not in chars_to_ignore:
                vocab.add(char)
    
    # Create vocabulary mapping
    vocab_list = sorted(list(vocab))
    vocab_dict = {char: idx for idx, char in enumerate(vocab_list)}
    
    # Add special tokens
    vocab_dict["<pad>"] = len(vocab_dict)
    vocab_dict["<unk>"] = len(vocab_dict)
    vocab_dict["|"] = len(vocab_dict)  # CTC blank token
    
    return vocab_dict

def load_data(
    train_data_path: str,
    valid_data_path: str,
    train_labels_path: str,
    valid_labels_path: str,
    chars_to_ignore: List[str] = None
) -> tuple:
    """Load training and validation data."""
    
    # Load audio file paths
    train_df = pd.read_csv(train_data_path, sep='\t', header=None, names=['path', 'frames'])
    valid_df = pd.read_csv(valid_data_path, sep='\t', header=None, names=['path', 'frames'])
    
    # Load labels
    with open(train_labels_path, 'r', encoding='utf-8') as f:
        train_labels = [line.strip() for line in f.readlines()]
    
    with open(valid_labels_path, 'r', encoding='utf-8') as f:
        valid_labels = [line.strip() for line in f.readlines()]
    
    # Create full paths
    base_path = Path(train_data_path).parent
    train_audio_paths = [str(base_path / path) for path in train_df['path']]
    valid_audio_paths = [str(base_path / path) for path in valid_df['path']]
    
    return train_audio_paths, valid_audio_paths, train_labels, valid_labels

class CTCTrainer(Trainer):
    """Custom Trainer for CTC loss computation."""
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Compute CTC loss
        loss = self.compute_ctc_loss(logits, labels)
        
        return (loss, outputs) if return_outputs else loss
    
    def compute_ctc_loss(self, logits, labels):
        """Compute CTC loss manually."""
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        input_lengths = torch.sum(labels != -100, dim=-1)
        target_lengths = torch.sum(labels != -100, dim=-1)
        
        # CTC loss
        ctc_loss = torch.nn.functional.ctc_loss(
            log_probs.transpose(0, 1),  # (T, N, C)
            labels,
            input_lengths,
            target_lengths,
            blank=0,  # Assuming blank token is at index 0
            reduction='mean'
        )
        
        return ctc_loss

def main():
    parser = argparse.ArgumentParser(description="Finetune Wav2Vec2 for Bengali ASR")
    parser.add_argument("--model_name", type=str, default="facebook/wav2vec2-base",
                       help="Hugging Face model name")
    parser.add_argument("--output_dir", type=str, default="./wav2vec2-bengali-asr",
                       help="Output directory for the model")
    parser.add_argument("--train_data_path", type=str, default="./files/train.tsv",
                       help="Path to training data")
    parser.add_argument("--valid_data_path", type=str, default="./files/valid.tsv",
                       help="Path to validation data")
    parser.add_argument("--train_labels_path", type=str, default="./train.ltr",
                       help="Path to training labels")
    parser.add_argument("--valid_labels_path", type=str, default="./train.ltr",
                       help="Path to validation labels")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                       help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--max_duration", type=float, default=20.0,
                       help="Maximum audio duration in seconds")
    parser.add_argument("--min_duration", type=float, default=1.0,
                       help="Minimum audio duration in seconds")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                       help="Number of gradient accumulation steps")
    parser.add_argument("--warmup_steps", type=int, default=500,
                       help="Number of warmup steps")
    parser.add_argument("--save_steps", type=int, default=500,
                       help="Number of steps between saves")
    parser.add_argument("--eval_steps", type=int, default=500,
                       help="Number of steps between evaluations")
    parser.add_argument("--logging_steps", type=int, default=100,
                       help="Number of steps between logging")
    parser.add_argument("--freeze_feature_extractor", action="store_true",
                       help="Freeze the feature extractor")
    parser.add_argument("--fp16", action="store_true",
                       help="Use mixed precision training")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info("Loading data...")
    train_audio_paths, valid_audio_paths, train_labels, valid_labels = load_data(
        args.train_data_path,
        args.valid_data_path,
        args.train_labels_path,
        args.valid_labels_path
    )
    
    # Create vocabulary
    logger.info("Creating vocabulary...")
    chars_to_ignore = [",", "?", ".", "!", "-", ";", ":", '""', "'", " "]
    vocab = create_vocab_from_labels(train_labels + valid_labels, chars_to_ignore)
    
    # Save vocabulary
    vocab_path = Path(args.output_dir) / "vocab.json"
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    
    # Create tokenizer
    logger.info("Creating tokenizer...")
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_file=str(vocab_path),
        unk_token="<unk>",
        pad_token="<pad>",
        word_delimiter_token="|"
    )
    
    # Create feature extractor
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True
    )
    
    # Create processor
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer
    )
    
    # Load model
    logger.info(f"Loading model: {args.model_name}")
    model = Wav2Vec2ForCTC.from_pretrained(
        args.model_name,
        vocab_size=len(vocab),
        ctc_loss_reduction="mean",
        pad_token_id=tokenizer.pad_token_id,
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.0,
        layerdrop=0.1,
        ctc_zero_infinity=True
    )
    
    # Freeze feature extractor if requested
    if args.freeze_feature_extractor:
        model.freeze_feature_extractor()
        logger.info("Feature extractor frozen")
    
    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = BengaliASRDataset(
        audio_paths=train_audio_paths,
        labels=train_labels,
        processor=processor,
        chars_to_ignore=chars_to_ignore,
        max_duration=args.max_duration,
        min_duration=args.min_duration
    )
    
    valid_dataset = BengaliASRDataset(
        audio_paths=valid_audio_paths,
        labels=valid_labels,
        processor=processor,
        chars_to_ignore=chars_to_ignore,
        max_duration=args.max_duration,
        min_duration=args.min_duration
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=args.fp16,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to=None,
    )
    
    # Create trainer
    trainer = CTCTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=processor.feature_extractor,
    )
    
    # Add early stopping
    early_stopping = EarlyStoppingCallback(early_stopping_patience=3)
    trainer.add_callback(early_stopping)
    
    # Start training
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model
    logger.info("Saving final model...")
    trainer.save_model()
    processor.save_pretrained(args.output_dir)
    
    logger.info(f"Training completed! Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
