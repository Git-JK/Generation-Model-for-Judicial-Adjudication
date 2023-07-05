PRE_SEQ_LEN=128
CHECKPOINT=adgen-chatglm-6b-pt-128-2e-2
STEP=3000

CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --do_predict \
    --validation_file AdvertiseGen/dev.json \
    --test_file AdvertiseGen/dev.json \
    --overwrite_cache \
    --prompt_column content \
    --response_column summary \
    --model_name_or_path THUDM/chatglm-6b \
    --ptuning_checkpoint ./output/$CHECKPOINT/checkpoint-$STEP \
    --output_dir ./output/$CHECKPOINT \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 64 \
    --per_device_eval_batch_size 1 \
    --predict_with_generate \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4

    # python main.py --do_predict --preprocessing_num_workers 8 --validation_file ../new_data/test.json --test_file ../new_data/test.json --overwrite_cache --prompt_column content --response_column summary --model_name_or_path THUDM/chatglm-6b --ptuning_checkpoint ../output/adgen-chatglm-6b-pt-128-0.02/checkpoint-19126 --output_dir ../output/adgen-chatglm-6b-pt-128-0.02 --overwrite_output_dir --max_source_length 1408 --max_target_length 512 --per_device_eval_batch_size 1 --predict_with_generate --pre_seq_len 128 --quantization_bit 4 --local_rank -1