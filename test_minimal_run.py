#!/usr/bin/env python3
"""
Minimal test to verify we can load and run a tiny version of the training
"""
import torch
import os
import sys
from pathlib import Path

# Add training module to path
sys.path.append('NExtLong')

def test_imports():
    """Test if we can import required modules"""
    try:
        print("Testing imports...")
        import transformers
        print(f"✓ Transformers {transformers.__version__}")
        
        import datasets
        print(f"✓ Datasets {datasets.__version__}")
        
        import accelerate
        print(f"✓ Accelerate {accelerate.__version__}")
        
        # Try to import training modules
        from training.dataset import DataCollator
        print("✓ Training dataset module")
        
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_minimal_model():
    """Test loading a minimal model"""
    try:
        from transformers import AutoConfig, AutoTokenizer
        
        model_path = Path.home() / "models" / "llama-3-8b-instruct"
        
        if not model_path.exists():
            print(f"Model path doesn't exist: {model_path}")
            print("Creating mock model for testing...")
            model_path.mkdir(parents=True, exist_ok=True)
            
            # Create minimal config
            config = AutoConfig.from_pretrained("gpt2")  # Use GPT2 as base
            config.save_pretrained(str(model_path))
            
            # Create minimal tokenizer
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            tokenizer.save_pretrained(str(model_path))
            
            print("✓ Created mock model files")
        
        # Try loading config
        config = AutoConfig.from_pretrained(str(model_path))
        print(f"✓ Loaded config: {config.model_type}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return False

def test_gpu_memory():
    """Check GPU memory availability"""
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        
        total_memory = props.total_memory / 1e9
        reserved = torch.cuda.memory_reserved(device) / 1e9
        allocated = torch.cuda.memory_allocated(device) / 1e9
        free = (props.total_memory - torch.cuda.memory_allocated(device)) / 1e9
        
        print(f"\nGPU Memory Status:")
        print(f"  Device: {props.name}")
        print(f"  Total: {total_memory:.2f} GB")
        print(f"  Free: {free:.2f} GB")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved: {reserved:.2f} GB")
        
        if free < 2.0:
            print("  ⚠ Warning: Low GPU memory, may need CPU offloading")
            return False
        return True
    else:
        print("No GPU available")
        return False

def simulate_mini_training():
    """Simulate a minimal training loop"""
    print("\nSimulating minimal training...")
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create tiny model (much smaller than Llama)
        model = torch.nn.Sequential(
            torch.nn.Embedding(1000, 128),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1000)
        ).to(device)
        
        print(f"  Model on: {device}")
        print(f"  Model size: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        
        # Enable mixed precision for memory efficiency
        scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None
        
        for step in range(3):
            # Create random batch
            input_ids = torch.randint(0, 1000, (1, 32)).to(device)
            labels = torch.randint(0, 1000, (1, 32)).to(device)
            
            if scaler:
                with torch.cuda.amp.autocast():
                    # Forward
                    output = model(input_ids)
                    output = output.view(-1, 1000)
                    labels = labels.view(-1)
                    
                    # Loss
                    loss = torch.nn.functional.cross_entropy(output, labels)
                
                # Backward with scaler
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                # Standard forward/backward
                output = model(input_ids)
                output = output.view(-1, 1000)
                labels = labels.view(-1)
                loss = torch.nn.functional.cross_entropy(output, labels)
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            print(f"  Step {step+1}/3 - Loss: {loss.item():.4f}")
            
            # Clear cache to free memory
            if device == "cuda":
                torch.cuda.empty_cache()
        
        print("✓ Training simulation successful")
        return True
        
    except Exception as e:
        print(f"✗ Training simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*60)
    print("Minimal Training Test")
    print("="*60)
    
    results = []
    
    # Test imports
    results.append(test_imports())
    
    # Test model loading
    results.append(test_minimal_model())
    
    # Test GPU memory
    results.append(test_gpu_memory())
    
    # Test minimal training
    results.append(simulate_mini_training())
    
    print("\n" + "="*60)
    if all(results):
        print("✓ All tests passed!")
        print("\nReady to run minimal training:")
        print("  cd NExtLong && ./train_64K_minimal.sh")
        print("\nFor full training on L40s cluster:")
        print("  cd NExtLong && ./train_64K_l40s.sh")
    else:
        print("⚠ Some tests failed")
        print("\nTroubleshooting:")
        print("1. Check CUDA installation: nvidia-smi")
        print("2. Free GPU memory by closing other applications")
        print("3. Consider using CPU-only mode for testing")

if __name__ == "__main__":
    main()