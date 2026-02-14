#!/bin/bash

# Random baseline for switch/flip ablation study - flips N random examples
# instead of flipping top N ranked examples

export NCCL_DEBUG=WARN
export HF_DATASETS_CACHE=$HOME/.cache/huggingface/datasets
export HF_HUB_CACHE=$HOME/.cache/huggingface/hub
RANDOM_N_VALUES=(
    30000
)

for RANDOM_N in "${RANDOM_N_VALUES[@]}"; do
    echo "=========================================="
    echo "Running training with random_filter_n=${RANDOM_N} (flip)"
    echo "=========================================="

    accelerate launch \
        --num_machines 1 \
        --num_processes 8 \
        --mixed_precision bf16 \
        --use_deepspeed \
        --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
        open_instruct/dpo_tune_cache.py \
        --exp_name olmo2_7b_dpo_switch_random_${RANDOM_N} \
        --model_name_or_path allenai/OLMo-2-1124-7B-SFT \
        --model_revision main \
        --tokenizer_name allenai/OLMo-2-1124-7B-SFT \
        --tokenizer_revision main \
        --use_slow_tokenizer False \
        --add_bos \
        --dataset_mixer_list allenai/olmo-2-1124-7b-preference-mix 1.0 \
        --max_seq_length 2048 \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 8 \
        --learning_rate 1e-6 \
        --lr_scheduler_type linear \
        --warmup_ratio 0.1 \
        --weight_decay 0.0 \
        --num_train_epochs 1 \
        --logging_steps 1 \
        --dpo_loss_type dpo_norm \
        --dpo_beta 5 \
        --use_flash_attn \
        --random_filter_n ${RANDOM_N} \
        --random_filter_action flip \
        --seed 63 \
        --checkpointing_steps 500 \
        --keep_last_n_checkpoints 50 \
        --max_train_samples 1000000 \
        --add_seed_and_date_to_exp_name False \
        --do_not_randomize_output_dir True \
        --push_to_hub False \
        --try_launch_beaker_eval_jobs False \
        --with_tracking \
        --output_dir output/olmo2_7b_dpo_switch_random_${RANDOM_N}_63

    echo ""
    echo "Completed training with random_filter_n=${RANDOM_N} (flip)"
    echo ""
done

echo "=========================================="
echo "All random switch training runs completed!"
echo "=========================================="
