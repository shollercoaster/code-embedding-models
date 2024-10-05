cd code
data_dir=/DS/dsg-ml/work/schaturv/code-embedding-models/text-code
python run_lora.py \
    --output_dir=$data_dir/saved_models \
    --model_type=roberta \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=roberta-base \
    --do_train \
    --train_data_file=$data_dir/dataset/train.jsonl \
    --eval_data_file=$data_dir/dataset/valid.jsonl \
    --test_data_file=$data_dir/dataset/test.jsonl \
    --epoch 1 \
    --block_size 256 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee train.log
