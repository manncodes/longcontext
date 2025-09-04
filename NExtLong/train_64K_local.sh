#!/bin/bash

# Local training script for single GPU with minimal resources

# Path to your local model checkpoint directory (HF format)
model=${MODEL:-~/models/llama-3-8b-instruct}
# Path to the local dataset directory
dataset=${DATASET:-~/datasets/nextlong-64k}

# Create minimal dataset structure if needed
mkdir -p $dataset

# Minimal domains for testing - using only one domain with tiny proportion
domains=(
    fineweb-edu@1.0
)
domains_name=ProLong64K_minimal

# Minimal batch size for 1650Ti (4GB VRAM)
bsz=${BSZ:-1} # Extremely small batch
seq=${SEQ:-1} # per-device batch size
lr=${LR:-1e-5}
steps=${STEPS:-5} # Just 5 steps for testing
save_steps=${SAVE:-5}
warmup=${WARMUP:-0.1}
suffix=${SUFFIX:-"_local_test"}

run_name="lcft_local_test_bsz${bsz}_steps${steps}_lr${lr}${suffix}"
out_dir="checkpoints/$run_name"

# Single GPU setup
num_gpus=1
num_nodes=1

# No gradient accumulation needed with single sample
accu=1

# Use NO_SHARD (DDP) for single GPU
fsdp=3
gc=1 # Enable gradient checkpointing to save memory

# Reduce logit block size for memory constraints
export LOGIT_BLOCK_SIZE=512

mkdir -p $out_dir
nvidia-smi

# Find available port
master_port=$(comm -23 <(seq 49152 65535 | sort) <(ss -Htan 2>/dev/null | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

# Single GPU launch
header="python -m training.train_language_model"

echo "Running locally with num_gpus=${num_gpus}"

export OMP_NUM_THREADS=1
export WANDB_PROJECT="prolong-local"
export WANDB_DIR=$out_dir
export WANDB_MODE="offline"
export TOKENIZERS_PARALLELISM=true

# Limit memory usage for 1650Ti
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

base_arguments=(
    --report_to wandb
    --do_train

    --model_name $model
    --tokenizer_name $model

    --run_name $run_name
    --output_dir $out_dir
    --gradient_accumulation_steps $accu
    --per_device_train_batch_size $seq
    --per_device_eval_batch_size $seq

    # Use fp16 instead of bf16 for older GPU
    --fp16
    --learning_rate $lr
    --min_lr_ratio 0.1
    --lr_scheduler_type cosine
    --max_grad_norm 1.0
    --adam_beta1 0.9
    --adam_beta2 0.95
    --weight_decay 0.1
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
    --ddp_find_unused_parameters false

    # Reduced sequence length for memory
    --per_device_max_tokens 2048

    --cuda_empty_cache
    --config_overrides "rope_theta=8000000"
)

# Enable gradient checkpointing
base_arguments+=( --gradient_checkpointing )

# Add tokenized dataset paths (simplified for testing)
base_arguments+=( --tokenized_mds_train )
for domain in "${domains[@]}"; do
    base_arguments+=( $dataset/$domain )
done

base_arguments+=( $@ )

echo "Command: ${header} ${base_arguments[@]}"
${header} "${base_arguments[@]}" 2>&1 | tee -a $out_dir/log.out