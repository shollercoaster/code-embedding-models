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
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1
    )
```
### Variables to change during training
- [x] LoRA rank and alpha (alpha / rank ratio is generally set to 2) (set rank to 8, alpha to 16)
- [x] batch_size overall = 64
- number of workers = 4
- model type (available options:` 'gpt2', 'openai-gpt', 'bert', 'roberta', 'distilbert'`)
- gradient accumulation steps (only helps with memory constraints)
- training optimizer config
- training scheduler config
- Learning rate
- [x] fp16 (currently not set)
- [x] multinode-multi GPU training using `DistributedDataParallel` (already happening)
- block_size (changes code token and padding length of both NL and PL)
## Apex command
`pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./` [this](https://github.com/huggingface/accelerate/pull/1689/commits/3cf93f773f2f625197da7c61e6ae640f86721a78) and [this references](https://github.com/NVIDIA/apex/pull/1690/commits/b34aedc7430a3b7671a4ec9ce0bdf83e5b6716ef)
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
## 2 epochs
![[Pasted image 20241007012712.png]]
- best MRR - 0.3389
- ran for 2hr 50 mins (without fp16, )
# 2 epochs 8 rank
```
10/08/2024 20:21:11 - INFO - __main__ -   ***** Running evaluation *****
10/08/2024 20:21:11 - INFO - __main__ -     Num examples = 9604
10/08/2024 20:21:11 - INFO - __main__ -     Batch size = 64
10/08/2024 20:22:40 - INFO - __main__ -     eval_loss = 0.8312
10/08/2024 20:22:40 - INFO - __main__ -     eval_mrr = 0.3263
```
best mrr = 0.3267
2hr50 mins
# 2 epochs 32 rank 
```
10/09/2024 12:50:43 - INFO - __main__ -   ***** Running evaluation *****
10/09/2024 12:50:43 - INFO - __main__ -     Num examples = 9604
10/09/2024 12:50:43 - INFO - __main__ -     Batch size = 64
10/09/2024 12:52:12 - INFO - __main__ -     eval_loss = 0.7731
10/09/2024 12:52:12 - INFO - __main__ -     eval_mrr = 0.3469
10/09/2024 12:52:12 - INFO - __main__ -     ********************
10/09/2024 12:52:12 - INFO - __main__ -     Best mrr:0.3469
```
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
