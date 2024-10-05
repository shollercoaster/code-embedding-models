Using the run.py script in the Siamese-model for finetuning Roberta model. But there was inconsistency in dataset and it did not have appropriate json fields (missing url fields, likely not from the CSN dataset). So I moved to CodeXGLUE
- Finally using training scripts inside codeXGLUE for different tasks. 
- Just need to understand how these scripts work, make my LoRA and language agnostic svd changes, and then i should have better mrr.
- Check target modules for LoRA for different transformer models [here](https://github.com/huggingface/peft/blob/632997d1fb776c3cf05d8c2537ac9a98a7ce9435/src/peft/utils/other.py#L202)
	- [ "query", "value ] for roberta.
- current config:
```Python
lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],  # RoBERTa attention layers to apply LoRA
        lora_dropout=0.1
    )
```
### Variables to change during training
- LoRA rank 
- LoRA alpha
- batch_size overall = 64
- number of workers = 4
- gradient accumulation steps
- training optimizer config
- training scheduler config
- Learning rate
- fp16 (currently not set)
- multinode-multi GPU training using `DistributedDataParallel` (already happening)
- block_size (changes code token and padding length of both NL and PL)
# Initial training (with LoRA) summary
```
10/05/2024 19:35:12 - INFO - ***** Running training *****
10/05/2024 19:35:12 - INFO -  Num examples = 251820
10/05/2024 19:35:12 - INFO -  Num Epochs = 1
10/05/2024 19:35:12 - INFO -  Instantaneous batch size per GPU = 32
10/05/2024 19:35:12 - INFO -  Total train batch size (w. parallel, distributed & accumulation) = 64
10/05/2024 19:35:12 - INFO -  Gradient Accumulation steps = 1
10/05/2024 19:35:12 - INFO -  Total optimization steps = 3935
trainable params: 589,824 || all params: 125,235,456 || trainable%: 0.4710
```
- first mrr jumped from 0.13 to double (0.25) in the second checkpoint only, but then increased only by 0.01 each checkpoint.
- eval loss decreased drastically from 15.7 to 0.176 by 3900 step
- took about 1hr25 mins for 1 epoch, 64 batch size with LoRA.
- best MRR is 0.3065 (max Python can get is 0.42)
# Initial training (without LoRA) summary
```
10/05/2024 17:14:35 -    ***** Running training *****
10/05/2024 17:14:35 -      Num examples = 251820
10/05/2024 17:14:35 -     Num Epochs = 2
10/05/2024 17:14:35 -      Instantaneous batch size per GPU = 16
10/05/2024 17:14:35 -      Total train batch size (w. parallel, distributed & accumulation) = 32
10/05/2024 17:14:35 -      Gradient Accumulation steps = 1
10/05/2024 17:14:35 -      Total optimization steps = 15740
```

- evaluation after 700 steps
- total 32 batches, 16 per GPU
- each epoch took 7k ish steps, so total steps - 15k ish (exact figure above)
- 5500 epochs ran for 1hr 15 mins 
```     
Num examples = 9604
10/05/2024 18:23:09 - INFO - __main__ -     Batch size = 64
10/05/2024 18:24:34 - INFO - __main__ -     eval_loss = 1.1528
10/05/2024 18:24:34 - INFO - __main__ -     eval_mrr = 0.2893
```
A singular evaluation
- eval loss decreased from 10.61472(step 100) to 1.1528 (step 4700)
- eval_mrr only changed from  0.2581 (step 700) to 0.2893 (step 4700)
- best mrr = 0.2893