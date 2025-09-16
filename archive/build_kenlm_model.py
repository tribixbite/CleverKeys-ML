#!/usr/bin/env python3
"""
Helper script to build a KenLM language model from vocabulary or corpus.
This creates a proper n-gram model for use with pyctcdecode.
"""

import os
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def build_kenlm_model_from_vocab(vocab_path: str, output_dir: str = "./kenlm_models"):
    """
    Build a simple KenLM model from a vocabulary file.
    This creates a unigram model which is better than nothing.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Read vocabulary
    logger.info(f"Reading vocabulary from {vocab_path}")
    with open(vocab_path, 'r') as f:
        words = [line.strip().lower() for line in f if line.strip()]
    
    logger.info(f"Found {len(words)} words")
    
    # Create a simple corpus with word frequencies
    corpus_file = os.path.join(output_dir, "corpus.txt")
    logger.info(f"Creating corpus file: {corpus_file}")
    
    with open(corpus_file, 'w') as f:
        # Write each word multiple times to create frequency distribution
        # More common words (shorter) get written more times
        for word in words:
            # Simple heuristic: shorter words are more common
            frequency = max(1, 10 - len(word))
            for _ in range(frequency):
                f.write(f"{word}\n")
    
    # Build ARPA file using KenLM
    arpa_file = os.path.join(output_dir, "vocab_lm.arpa")
    logger.info(f"Building ARPA language model: {arpa_file}")
    
    # Check if lmplz exists in kenlm/build/bin
    lmplz_path = "./kenlm/build/bin/lmplz"
    if not os.path.exists(lmplz_path):
        logger.error(f"lmplz not found at {lmplz_path}")
        logger.info("Please build KenLM first:")
        logger.info("  cd kenlm && mkdir -p build && cd build")
        logger.info("  cmake .. && make -j4")
        return None
    
    # Build 2-gram model (unigram + bigram)
    cmd = [
        lmplz_path,
        "-o", "2",  # 2-gram model
        "--discount_fallback",  # Use fallback discounting for small data
        "--text", corpus_file,
        "--arpa", arpa_file
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"✓ ARPA model created: {arpa_file}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to build ARPA model: {e}")
        logger.error(f"stderr: {e.stderr}")
        return None
    
    # Build binary model for faster loading
    binary_file = os.path.join(output_dir, "vocab_lm.bin")
    build_binary_path = "./kenlm/build/bin/build_binary"
    
    if os.path.exists(build_binary_path):
        logger.info(f"Building binary model: {binary_file}")
        cmd = [build_binary_path, arpa_file, binary_file]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"✓ Binary model created: {binary_file}")
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to build binary model: {e}")
            logger.info("You can still use the ARPA model")
    
    return arpa_file

def download_pretrained_model():
    """
    Download a pre-trained English language model.
    """
    logger.info("Downloading pre-trained language models...")
    
    # Option 1: Small Wikipedia-based model
    from huggingface_hub import hf_hub_download
    
    try:
        # Download a small English LM from Hugging Face
        lm_path = hf_hub_download(
            repo_id="edugp/kenlm", 
            filename="en/en.arpa.bin",
            cache_dir="./kenlm_models"
        )
        logger.info(f"✓ Downloaded pre-trained model to: {lm_path}")
        return lm_path
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        return None

def main():
    """Build or download a KenLM model for the training script."""
    
    logger.info("=" * 60)
    logger.info("KenLM Language Model Builder")
    logger.info("=" * 60)
    
    # Check if vocabulary file exists
    vocab_path = "vocab/final_vocab.txt"
    
    if os.path.exists(vocab_path):
        logger.info(f"Found vocabulary file: {vocab_path}")
        
        # Option 1: Build from vocabulary
        logger.info("\nOption 1: Building KenLM model from vocabulary...")
        arpa_path = build_kenlm_model_from_vocab(vocab_path)
        
        if arpa_path:
            logger.info("\n✅ Successfully built language model!")
            logger.info(f"ARPA model: {arpa_path}")
            logger.info("\nTo use this model, update your train.py config:")
            logger.info("  Add this path to the 'potential_models' list")
    else:
        logger.warning(f"Vocabulary file not found: {vocab_path}")
    
    # Option 2: Download pre-trained model
    logger.info("\nOption 2: Downloading pre-trained model...")
    pretrained_path = download_pretrained_model()
    
    if pretrained_path:
        logger.info("\n✅ Downloaded pre-trained model!")
        logger.info(f"Model path: {pretrained_path}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Setup complete! Your language models are ready.")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
