PRE_SEQ_LEN=128
LR=2e-2

CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --do_train \
    --train_file ../data/train.json \
    --validation_file ../data/valid.json \
    --prompt_column content \
    --response_column summary \
    --overwrite_cache \
    --model_name_or_path THUDM/chatglm-6b \
    --output_dir ../output \
    --overwrite_output_dir \
    --max_source_length 1408 \
    --max_target_length 512 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 4\
    --predict_with_generate \
    --logging_steps 2000 \
    --save_strategy epoch \
    --learning_rate 0.02 \
    --pre_seq_len 128 \
    --quantization_bit 4

# python main.py --do_train --train_file ../new_data/train.json --validation_file ../new_data/valid.json --prompt_column content --response_column summary --overwrite_cache --model_name_or_path THUDM/chatglm-6b --output_dir ../output/adgen-chatglm-6b-pt-128-0.02 --overwrite_output_dir --max_source_length 1408 --max_target_length 512 --per_device_train_batch_size 4 --per_device_eval_batch_size 1 --gradient_accumulation_steps 1 --predict_with_generate --num_train_epochs 2 --logging_steps 2000 --save_strategy epoch --learning_rate 0.02 --pre_seq_len 128 --quantization_bit 4