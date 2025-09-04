#!/bin/bash

# Ultra-minimal training script for 4GB GPU (1650Ti)
# Uses CPU offloading and extreme optimizations

# Path to your local model checkpoint directory (HF format)
model=${MODEL:-~/models/llama-3-8b-instruct}
# Path to the local dataset directory
dataset=${DATASET:-~/datasets/nextlong-64k}

# Create minimal dataset structure if needed
mkdir -p $dataset

# Single domain for minimal testing
domains=(
    fineweb-edu@1.0
)
domains_name=minimal_test

# Ultra-minimal settings for 4GB GPU
bsz=${BSZ:-1}
seq=${SEQ:-1} 
lr=${LR:-1e-5}
steps=${STEPS:-3} # Just 3 steps
save_steps=${SAVE:-3}
warmup=${WARMUP:-0}
suffix=${SUFFIX:-"_minimal"}

run_name="lcft_minimal_test_steps${steps}${suffix}"
out_dir="checkpoints/$run_name"

# Single GPU
num_gpus=1
num_nodes=1
accu=1

# CPU offloading mode
fsdp=0  # Disable FSDP
gc=1    # Enable gradient checkpointing

# Minimal memory settings
export LOGIT_BLOCK_SIZE=256
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64

mkdir -p $out_dir

# Check available GPU memory
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits

# Simple python launch (no torchrun for single GPU)
header="python -m training.train_language_model"

echo "Running minimal test with 4GB GPU constraints"

export OMP_NUM_THREADS=1
export WANDB_MODE="disabled" # Disable wandb completely for testing
export TOKENIZERS_PARALLELISM=false

base_arguments=(
    --do_train

    --model_name $model
    --tokenizer_name $model

    --run_name $run_name
    --output_dir $out_dir
    --gradient_accumulation_steps $accu
    --per_device_train_batch_size $seq
    --per_device_eval_batch_size $seq

    # Memory optimizations
    --fp16  # Use FP16
    --fp16_full_eval false
    --fp16_backend "auto"
    
    # Training settings
    --learning_rate $lr
    --min_lr_ratio 0.1
    --lr_scheduler_type constant
    --max_grad_norm 1.0
    --adam_beta1 0.9
    --adam_beta2 0.95
    --weight_decay 0.0
    --warmup_ratio $warmup
    --optim adamw_torch

    --logging_steps 1
    --log_level info

    --max_steps $steps
    --save_steps $save_steps
    --dataloader_num_workers 0

    --disable_tqdm false
    --use_fast_tokenizer false
    --remove_unused_columns false
    
    # Extreme memory reduction
    --per_device_max_tokens 512  # Very short sequences
    
    --gradient_checkpointing  # Must enable
    --cuda_empty_cache
    
    # Disable unnecessary features
    --load_best_model_at_end false
    --metric_for_best_model "loss"
    --greater_is_better false
    --save_total_limit 1
    --save_strategy "steps"
    --evaluation_strategy "no"
    --eval_steps 999999
    --report_to "none"
)

# Add dataset paths
base_arguments+=( --tokenized_mds_train )
for domain in "${domains[@]}"; do
    base_arguments+=( $dataset/$domain )
done

base_arguments+=( $@ )

echo ""
echo "Command: ${header} ${base_arguments[@]}"
echo ""
echo "Note: This is configured for extreme memory constraints (4GB GPU)"
echo "- Sequence length: 512 tokens (vs 65536 normal)"
echo "- Batch size: 1"
echo "- Gradient checkpointing: enabled"
echo "- FP16 precision"
echo "- Only 3 training steps"
echo ""

# Run with error catching
${header} "${base_arguments[@]}" 2>&1 | tee -a $out_dir/log.out || {
    echo ""
    echo "Training failed. Common issues:"
    echo "1. Out of memory - try reducing per_device_max_tokens further (256 or 128)"
    echo "2. Missing model files - ensure model is downloaded to $model"
    echo "3. Dataset format issues - check dataset structure"
    echo ""
    echo "For CPU-only training, add: --no_cuda"
}