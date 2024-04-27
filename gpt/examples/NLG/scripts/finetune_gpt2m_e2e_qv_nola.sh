PEFT=NOLA
NUM_BASIS=512 
RANK=8
LORA_ALPHA=8 
LR=0.005

python -m torch.distributed.launch --master_port 9091 --nproc_per_node=1 src/gpt2_ft.py \
    --train_data ./data/e2e/train.jsonl \
    --valid_data ./data/e2e/valid.jsonl \
    --log_interval 500 \
    --eval_interval 20000 \
    --train_batch_size 8 \
    --grad_acc 1 \
    --valid_batch_size 4 \
    --seq_len 512 \
    --model_card gpt2.md \
    --init_checkpoint ./pretrained_checkpoints/gpt2-medium-pytorch_model.bin \
    --platform local \
    --clip 0.0 \
    --weight_decay 0.0 \
    --correct_bias \
    --adam_beta2 0.999 \
    --scheduler linear \
    --warmup_step 500 \
    --max_epoch 5 \
    --save_interval 20000 \
    --label_smooth 0.1 \
    --lora_qv \
    --use_nola \
    --nola_num_basis $NUM_BASIS \
    --lora_rank $RANK \
    --lora_alpha $LORA_ALPHA \
    --lr $LR \
    --work_dir /nfs_share3/code/soroush/checkpoints/trained_models/GPT2_M/e2e_qv_${PEFT}_lr${LR}_rank${RANK}_alpha${LORA_ALPHA}_nbasis${NUM_BASIS} \
    --random_seed 110
    
    
    
python -m torch.distributed.launch --master_port $1 --nproc_per_node=1 src/gpt2_beam.py \
    --data ./data/e2e/test.jsonl \
    --batch_size 1 \
    --seq_len 512 \
    --eval_len 64 \
    --model_card gpt2.md \
    --init_checkpoint /nfs_share3/code/soroush/checkpoints/trained_models/GPT2_M/e2e_qv_${PEFT}_lr${LR}_rank${RANK}_alpha${LORA_ALPHA}_nbasis${NUM_BASIS}/model.26290.pt \
    --platform local \
    --lora_qv \
    --use_nola \
    --nola_num_basis $NUM_BASIS \
    --lora_rank $RANK \
    --lora_alpha $LORA_ALPHA \
    --beam 10 \
    --length_penalty 0.8 \
    --no_repeat_ngram_size 4 \
    --repetition_penalty 1.0 \
    --eos_token_id 628 \
    --work_dir /nfs_share3/code/soroush/checkpoints/trained_models/GPT2_M/e2e_qv_${PEFT}_lr${LR}_rank${RANK}_alpha${LORA_ALPHA}_nbasis${NUM_BASIS} \
    --output_file predict.26290.b10p08r4.jsonl
    
    
python src/gpt2_decode.py     --vocab ./vocab     --sample_file /nfs_share3/code/soroush/checkpoints/trained_models/GPT2_M/e2e_qv_${PEFT}_lr${LR}_rank${RANK}_alpha${LORA_ALPHA}_nbasis${NUM_BASIS}/predict.26290.b10p08r4.jsonl     --input_file ./data/e2e/test_formatted.jsonl     --output_ref_file e2e_ref.txt     --output_pred_file e2e_pred_${PEFT}${LR}${RANK}${LORA_ALPHA}${NUM_BASIS}.txt

python eval/e2e/measure_scores.py e2e_ref.txt e2e_pred_${PEFT}${LR}${RANK}${LORA_ALPHA}${NUM_BASIS}.txt -p



