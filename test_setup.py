#!/usr/bin/env python3
"""
Test script to verify the setup works with minimal resources
"""
import os
import sys
import torch
import json
from pathlib import Path

def check_gpu():
    """Check GPU availability"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✓ GPU found: {gpu_name} ({gpu_memory:.1f} GB)")
        return True
    else:
        print("✗ No GPU found - will use CPU")
        return False

def check_directories():
    """Check if required directories exist"""
    model_dir = Path.home() / "models" / "llama-3-8b-instruct"
    dataset_dir = Path.home() / "datasets" / "nextlong-64k"
    
    checks = []
    
    if model_dir.exists():
        print(f"✓ Model directory exists: {model_dir}")
        if (model_dir / "config.json").exists():
            print("  ✓ Model config found")
            checks.append(True)
        else:
            print("  ✗ Model config missing")
            checks.append(False)
    else:
        print(f"✗ Model directory missing: {model_dir}")
        checks.append(False)
    
    if dataset_dir.exists():
        print(f"✓ Dataset directory exists: {dataset_dir}")
        checks.append(True)
    else:
        print(f"✗ Dataset directory missing: {dataset_dir}")
        checks.append(False)
    
    return all(checks)

def test_model_loading():
    """Test if we can load model config"""
    try:
        from transformers import AutoConfig
        model_path = Path.home() / "models" / "llama-3-8b-instruct"
        
        if (model_path / "config.json").exists():
            config = AutoConfig.from_pretrained(str(model_path))
            print(f"✓ Model config loaded successfully")
            print(f"  Model type: {config.model_type}")
            print(f"  Hidden size: {config.hidden_size}")
            print(f"  Num layers: {config.num_hidden_layers}")
            return True
        else:
            print("✗ Model config not found")
            return False
    except Exception as e:
        print(f"✗ Error loading model config: {e}")
        return False

def test_memory_estimation():
    """Estimate memory requirements"""
    if torch.cuda.is_available():
        free_memory = torch.cuda.mem_get_info()[0] / 1e9
        total_memory = torch.cuda.mem_get_info()[1] / 1e9
        print(f"\nMemory Status:")
        print(f"  Total GPU memory: {total_memory:.1f} GB")
        print(f"  Free GPU memory: {free_memory:.1f} GB")
        
        # Estimate for 8B model with minimal batch
        model_memory = 8 * 2  # 8B params * 2 bytes (fp16)
        activation_memory = 2  # Rough estimate for batch=1, seq=2048
        total_needed = model_memory + activation_memory
        
        print(f"  Estimated memory needed: ~{total_needed:.1f} GB")
        
        if free_memory < total_needed:
            print(f"  ⚠ Warning: May need to reduce model size or use CPU offloading")
            return False
        else:
            print(f"  ✓ Sufficient memory available")
            return True
    else:
        print("No GPU available for memory check")
        return False

def create_mini_test():
    """Create a minimal test that simulates training"""
    print("\n" + "="*60)
    print("Running minimal training simulation...")
    print("="*60)
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Create tiny model for testing
        model = torch.nn.Linear(100, 100).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        
        # Simulate training steps
        for step in range(5):
            # Random input
            x = torch.randn(1, 100).to(device)
            
            # Forward pass
            output = model(x)
            loss = output.mean()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f"  Step {step+1}/5 - Loss: {loss.item():.4f}")
        
        print("✓ Training simulation completed successfully")
        return True
        
    except Exception as e:
        print(f"✗ Training simulation failed: {e}")
        return False

def main():
    print("="*60)
    print("NextLong Training Setup Test")
    print("="*60)
    
    results = []
    
    print("\n1. Checking GPU...")
    results.append(check_gpu())
    
    print("\n2. Checking directories...")
    results.append(check_directories())
    
    print("\n3. Testing model loading...")
    results.append(test_model_loading())
    
    print("\n4. Checking memory requirements...")
    results.append(test_memory_estimation())
    
    print("\n5. Running training simulation...")
    results.append(create_mini_test())
    
    print("\n" + "="*60)
    print("Test Summary:")
    print("="*60)
    
    if all(results):
        print("✓ All tests passed! Ready for training.")
        print("\nNext steps:")
        print("1. Download actual model: huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct --local-dir ~/models/llama-3-8b-instruct")
        print("2. Download NextLong dataset to ~/datasets/nextlong-64k/")
        print("3. Run: cd NExtLong && ./train_64K_local.sh")
    else:
        print("⚠ Some tests failed. Please check the issues above.")
        print("\nQuick fixes:")
        print("- Ensure CUDA is properly installed")
        print("- Run: python3 setup_local.py (to create directories)")
        print("- Check GPU memory usage: nvidia-smi")
    
    return 0 if all(results) else 1

if __name__ == "__main__":
    sys.exit(main())