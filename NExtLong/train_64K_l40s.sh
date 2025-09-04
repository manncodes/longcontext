#!/bin/bash

# Production training script for 8x L40s GPUs
# Optimized for L40s (48GB VRAM each)

# Path to your model checkpoint directory (HF format) 
model=${MODEL:-/path/to/llama-3-8b-instruct}
# Path to the dataset directory
dataset=${DATASET:-/path/to/nextlong-64k}

# Full domain configuration for production
domains=(
    thestackv1_concat_by_repo-65536@0.3
    book-65536@0.3
    fineweb-edu@0.1
    fineweb-2023-50@0.1
    stackexchange@0.04
    dolmawiki@0.04
    tuluv2@0.03
    arxiv@0.03
    openwebmath@0.03
    textbooks@0.03
)
domains_name=ProLong64KMix

# Optimized batch size for L40s (48GB VRAM)
bsz=${BSZ:-128} # Increased batch size for L40s
seq=${SEQ:-2} # per-device batch size
lr=${LR:-1e-5}
steps=${STEPS:-500}
save_steps=${SAVE:-125}
warmup=${WARMUP:-0.1}
suffix=${SUFFIX:-"_l40s"}

run_name="lcft_$(basename $model)_$(basename $dataset)_${domains_name}_bsz${bsz}_steps${steps}_lr${lr}_warmup${warmup}${suffix}"
out_dir="checkpoints/$run_name"

# 8 GPUs configuration
num_gpus=8
num_nodes=1

# Calculate gradient accumulation
accu=$(($bsz / $seq / $num_gpus / $num_nodes))

# Use FULL_SHARD for multi-GPU training
fsdp=1
gc=1 # Enable gradient checkpointing

# Optimized for L40s memory
export LOGIT_BLOCK_SIZE=4096

mkdir -p $out_dir
nvidia-smi

# Find available port
master_port=$(comm -23 <(seq 49152 65535 | sort) <(ss -Htan 2>/dev/null | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

# Multi-GPU launch with torchrun
header="torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:$master_port \
    --nnodes=1 \
    --nproc-per-node=$num_gpus \
    -m training.train_language_model"

echo "Running on L40s cluster with num_gpus=${num_gpus}"

export OMP_NUM_THREADS=$num_gpus
export WANDB_PROJECT="prolong-l40s"
export WANDB_DIR=$out_dir
export WANDB_MODE="offline"
export TOKENIZERS_PARALLELISM=true

# L40s optimizations
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0
export NCCL_P2P_DISABLE=0
export NCCL_TREE_THRESHOLD=0

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

    --bf16
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
    --dataloader_num_workers 4

    --disable_tqdm false
    --use_fast_tokenizer false
    --remove_unused_columns false
    --ddp_find_unused_parameters false

    --per_device_max_tokens 65536

    --torch_compile
    --cuda_empty_cache
    --config_overrides "rope_theta=8000000"
)

# FSDP configuration
if [ $fsdp -ne 0 ]; then
    export FSDP_SHARDING_STRATEGY=$fsdp 
    base_arguments+=( --fsdp "auto_wrap" )
    export FSDP_STATE_DICT_TYPE="FULL_STATE_DICT"
fi

# Gradient checkpointing
if [ $gc -ne 0 ]; then
    base_arguments+=( --gradient_checkpointing )
fi

# Add tokenized dataset paths
base_arguments+=( --tokenized_mds_train )
for domain in "${domains[@]}"; do
    base_arguments+=( $dataset/$domain )
done

base_arguments+=( $@ )

echo "Command: ${header} ${base_arguments[@]}"
${header} "${base_arguments[@]}" 2>&1 | tee -a $out_dir/log.out