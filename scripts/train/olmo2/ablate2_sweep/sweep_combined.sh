#!/bin/bash

export NCCL_DEBUG=WARN
TOP_N_VALUES=(
	0
)

for TOP_N in "${TOP_N_VALUES[@]}"; do
    
    accelerate launch \
        --num_machines 1 \
        --num_processes 8 \
        --mixed_precision bf16 \
        --use_deepspeed \
        --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
        open_instruct/dpo_tune_cache.py \
        --exp_name olmo2_7b_dpo_${TOP_N} \
        --model_name_or_path allenai/OLMo-2-1124-7B-SFT \
        --model_revision main \
        --tokenizer_name allenai/OLMo-2-1124-7B-SFT \
        --tokenizer_revision main \
        --use_slow_tokenizer False \
        --add_bos \
        --dataset_mixer_list allenai/olmo-2-1124-7b-preference-mix 1.0 \
        --max_seq_length 2048 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 16 \
        --learning_rate 1e-6 \
        --lr_scheduler_type linear \
        --warmup_ratio 0.1 \
        --weight_decay 0.0 \
        --num_train_epochs 1 \
        --logging_steps 1 \
        --dpo_loss_type dpo_norm \
        --dpo_beta 5 \
        --use_flash_attn \
        --gradient_checkpointing \
        --ranking_filter_jsonl /mnt/polished-lake/home/fxiao-two/LESS/outputs/top_17600_combined_union.jsonl \
        --ranking_filter_top_n 9999999 \
        --checkpointing_steps 500 \
        --keep_last_n_checkpoints 50 \
        --max_train_samples 1000000 \
        --add_seed_and_date_to_exp_name False \
        --do_not_randomize_output_dir True \
        --push_to_hub False \
        --try_launch_beaker_eval_jobs False \
        --with_tracking \
        --sample_before_filtering \
        --output_dir output/olmo2_7b_dpo_combined_30k
    
    echo ""
    echo "Completed training with ranking_filter_top_n=${TOP_N}"
    echo ""
done

echo "=========================================="
echo "All training runs completed!"
echo "=========================================="

