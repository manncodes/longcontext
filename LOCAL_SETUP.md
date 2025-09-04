# Local Setup Guide for NextLong Training

## Quick Start

### 1. Environment Setup
```bash
# Install minimal dependencies for testing
cd NExtLong
pip install -r requirements_minimal.txt

# For full dependencies (includes flash-attn, etc.)
pip install -r requirements.txt
```

### 2. Model Setup
Place your Hugging Face format model checkpoint in `~/models/llama-3-8b-instruct/`

```bash
# Option A: Download from Hugging Face
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct \
  --local-dir ~/models/llama-3-8b-instruct

# Option B: Use existing checkpoint
cp -r /path/to/your/model ~/models/llama-3-8b-instruct
```

### 3. Dataset Setup
Place NextLong 64K dataset in `~/datasets/nextlong-64k/`

The dataset should have the following structure:
```
~/datasets/nextlong-64k/
├── thestackv1_concat_by_repo-65536/
├── book-65536/
├── fineweb-edu/
├── fineweb-2023-50/
├── stackexchange/
├── dolmawiki/
├── tuluv2/
├── arxiv/
├── openwebmath/
└── textbooks/
```

### 4. Test Run (1650Ti - Minimal Resources)
```bash
cd NExtLong
./train_64K_local.sh
```

This script is configured for:
- Single GPU with 4GB VRAM
- Batch size of 1
- Sequence length of 2048 tokens (reduced from 65536)
- 5 training steps for testing
- FP16 precision (instead of BF16)
- Gradient checkpointing enabled

### 5. Production Run (8x L40s GPUs)
```bash
cd NExtLong
# Update paths in the script
export MODEL=/path/to/llama-3-8b-instruct
export DATASET=/path/to/nextlong-64k
./train_64K_l40s.sh
```

This script is optimized for:
- 8x L40s GPUs (48GB VRAM each)
- Batch size of 128
- Full 65536 token sequence length
- 500 training steps
- BF16 precision
- FSDP for distributed training
- Torch compilation for speed

## Script Modifications

### Key Changes for Local Testing (`train_64K_local.sh`):
- Reduced batch size: 64 → 1
- Reduced sequence length: 65536 → 2048
- Reduced training steps: 500 → 5
- Changed precision: BF16 → FP16
- Disabled multi-GPU: 8 GPUs → 1 GPU
- Disabled FSDP: Uses simple DDP
- Reduced dataloader workers: 1 → 0

### Optimizations for L40s (`train_64K_l40s.sh`):
- Increased batch size: 64 → 128
- Increased LOGIT_BLOCK_SIZE: 2048 → 4096
- Added NCCL optimizations
- Enabled torch compilation
- Increased dataloader workers: 1 → 4
- Optimized memory allocation settings

## Monitoring Training

Training logs will be saved to:
- Local test: `checkpoints/lcft_local_test_*/log.out`
- L40s: `checkpoints/lcft_*_l40s/log.out`

WandB logs (offline mode) will be saved in the same directories.

## Troubleshooting

### Out of Memory on 1650Ti
- Reduce `per_device_max_tokens` in `train_64K_local.sh` (try 1024 or 512)
- Ensure gradient checkpointing is enabled (`gc=1`)
- Close other GPU-using applications

### Missing Dependencies
```bash
# Install CUDA toolkit if needed
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia

# Install flash-attention (optional, for faster training)
pip install flash-attn --no-build-isolation
```

### Dataset Format Issues
The training script expects tokenized MDS format. If you have raw text, you'll need to preprocess it first using the chunking utilities in the repository.