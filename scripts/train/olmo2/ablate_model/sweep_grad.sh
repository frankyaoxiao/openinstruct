#!/bin/bash

export NCCL_DEBUG=WARN
accelerate launch \
    --num_machines 1 \
    --num_processes 8 \
    --mixed_precision bf16 \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
    open_instruct/dpo_tune_cache.py \
    --exp_name olmo2_7b_dpo_ablate_model \
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
    --ranking_filter_jsonl rankings_dpo.jsonl \
    --checkpointing_steps 500 \
    --keep_last_n_checkpoints 50 \
    --max_train_samples 1000000 \
    --exclude_chosen_models internlm/internlm2_5-1_8b-chat 01-ai/Yi-6B-Chat internlm/internlm2_5-7b-chat 01-ai/Yi-34B-Chat \
    --exclude_models_match_rejected \
    --add_seed_and_date_to_exp_name False \
    --do_not_randomize_output_dir True \
    --with_tracking \
    --output_dir output/olmo2_7b_dpo_ablate_model_grad
