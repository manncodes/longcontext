#!/usr/bin/env python3
"""
Setup script for local model and dataset preparation
"""
import os
import json
from pathlib import Path

def setup_directories():
    """Create necessary directories"""
    dirs = [
        Path.home() / "models" / "llama-3-8b-instruct",
        Path.home() / "datasets" / "nextlong-64k" / "fineweb-edu@1.0",
    ]
    
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {d}")
    
    return dirs[0], dirs[1].parent

def create_mock_model_config(model_dir):
    """Create minimal model config for testing"""
    config = {
        "architectures": ["LlamaForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 128000,
        "eos_token_id": 128001,
        "hidden_act": "silu",
        "hidden_size": 4096,
        "initializer_range": 0.02,
        "intermediate_size": 14336,
        "max_position_embeddings": 8192,
        "model_type": "llama",
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "num_key_value_heads": 8,
        "pretraining_tp": 1,
        "rms_norm_eps": 1e-05,
        "rope_scaling": None,
        "rope_theta": 500000.0,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.44.2",
        "use_cache": True,
        "vocab_size": 128256
    }
    
    config_path = model_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Created model config: {config_path}")
    
    # Create tokenizer config
    tokenizer_config = {
        "add_bos_token": True,
        "add_eos_token": False,
        "added_tokens_decoder": {},
        "bos_token": "<|begin_of_text|>",
        "clean_up_tokenization_spaces": True,
        "eos_token": "<|end_of_text|>",
        "model_input_names": ["input_ids", "attention_mask"],
        "model_max_length": 131072,
        "pad_token": None,
        "padding_side": "left",
        "tokenizer_class": "PreTrainedTokenizerFast"
    }
    
    tokenizer_path = model_dir / "tokenizer_config.json"
    with open(tokenizer_path, 'w') as f:
        json.dump(tokenizer_config, f, indent=2)
    print(f"Created tokenizer config: {tokenizer_path}")
    
def create_sample_dataset(dataset_dir):
    """Create a minimal sample dataset for testing"""
    import numpy as np
    
    # Create sample tokenized data
    sample_data = {
        "input_ids": np.random.randint(0, 128000, size=(10, 2048), dtype=np.int32),
        "attention_mask": np.ones((10, 2048), dtype=np.int32)
    }
    
    # Save as a simple format that can be loaded
    data_file = dataset_dir / "fineweb-edu@1.0" / "sample.npz"
    np.savez_compressed(data_file, **sample_data)
    print(f"Created sample dataset: {data_file}")
    
    # Create metadata
    metadata = {
        "dataset_name": "fineweb-edu-sample",
        "num_samples": 10,
        "max_length": 2048,
        "format": "numpy"
    }
    
    meta_file = dataset_dir / "fineweb-edu@1.0" / "metadata.json"
    with open(meta_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Created metadata: {meta_file}")

def download_instructions():
    """Print instructions for downloading actual model and dataset"""
    print("\n" + "="*60)
    print("SETUP COMPLETE - Next Steps:")
    print("="*60)
    
    print("\n1. Download your model checkpoint:")
    print("   Option A: Use Hugging Face CLI")
    print("   huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct --local-dir ~/models/llama-3-8b-instruct")
    print("\n   Option B: Use git clone")
    print("   cd ~/models && git clone https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct llama-3-8b-instruct")
    
    print("\n2. Download NextLong 64K dataset:")
    print("   The actual dataset needs to be downloaded from the official source")
    print("   Place it in ~/datasets/nextlong-64k/")
    
    print("\n3. Install dependencies:")
    print("   cd NExtLong && pip install -r requirements.txt")
    
    print("\n4. Run the training script:")
    print("   cd NExtLong && ./train_64K_local.sh")
    
    print("\nNote: The mock files created here are for testing the setup.")
    print("Replace them with actual model and dataset files for real training.")

def main():
    print("Setting up local environment for NextLong training...")
    
    model_dir, dataset_dir = setup_directories()
    create_mock_model_config(model_dir)
    create_sample_dataset(dataset_dir)
    download_instructions()

if __name__ == "__main__":
    main()