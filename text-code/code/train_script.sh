cd code
data_dir=/workspace/text-code
python run_lora.py \
    --output_dir=$data_dir/saved_models \
    --model_type=roberta \
    --config_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --do_train \
    --train_data_file=$data_dir/dataset/train.jsonl \
    --eval_data_file=$data_dir/dataset/valid.jsonl \
    --test_data_file=$data_dir/dataset/test.jsonl \
    --fp16_opt_level=O1 \
    --epoch 2 \
    --block_size 256 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee train.log
