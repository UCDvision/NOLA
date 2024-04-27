python qlora.py \
    --model_name_or_path /nfs_share3/code/soroush/LLM/Llama-2-70b-hf \
    --output_dir ./output/llama-2-guanaco-7b \
    --logging_steps 5 \
    --save_strategy steps \
    --data_seed 42 \
    --save_steps 500 \
    --save_total_limit 40 \
    --evaluation_strategy steps \
    --eval_dataset_size 1024 \
    --max_eval_samples 1000 \
    --per_device_eval_batch_size 1 \
    --max_new_tokens 32 \
    --dataloader_num_workers 8 \
    --group_by_length \
    --logging_strategy steps \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --do_mmlu_eval \
    --mmlu_split 'test'\
    --use_nola \
    --nola_num_basis 512 \
    --lora_r 16 \
    --lora_alpha 4 \
    --quant_type nf4 \
    --bf16 \
    --bits 4 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type constant \
    --gradient_checkpointing \
    --dataset alpaca \
    --source_max_len 384 \
    --target_max_len 512 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 128 \
    --max_steps 200 \
    --eval_steps 200 \
    --learning_rate 0.001 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.9 \
    --lora_dropout 0.0 \
    --weight_decay 0.0 \
    --seed 0 \


