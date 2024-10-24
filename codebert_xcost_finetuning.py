# -*- coding: utf-8 -*-
"""codebert-xcost-finetuning.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1pCqxCwNcC9vLbFe5zES373QWFu0lRgQm
"""


from transformers import RobertaTokenizer, RobertaModel
from transformers import TrainingArguments
import sys
import os
from code_search import get_dataset, collate_fn, ContrastiveTrainer
from retrieval_utils import calculate_mrr_at_step, construct_database_and_perform_search, finetune_with_mrr, get_rank_of_key
import torch
from torch.utils.data import DataLoader, SequentialSampler, DistributedSampler
import numpy as np

def get_model():
    model = RobertaModel.from_pretrained('microsoft/codebert-base')
    tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
    model = model.eval().cuda()
    return model, tokenizer

model, tokenizer = get_model()

languages = ["C", "PHP", "Java", "C++", "C#", "Javascript", "Python"]
root_path = "XLCoST_data"

dataset = get_dataset(root_path=root_path, languages=languages)

training_args = TrainingArguments(
    "contrastive_trainer",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_steps=200,
    num_train_epochs=1,
    evaluation_strategy="no",
    report_to="none",
    remove_unused_columns=False,
    warmup_steps=1000,
    save_strategy="epoch"
)
def evaluate(args, model, tokenizer, eval_when_training=False):
    eval_output_dir = args.output_dir
    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    # Create eval dataset and dataloader
    eval_dataset = dataset["val"]
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Multi-GPU support
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    print("***** Running evaluation *****")
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    code_vecs = []
    nl_vecs = []
    
    for batch in eval_dataloader:
        code_inputs = batch[0].to(args.device)
        nl_inputs = batch[1].to(args.device)
        with torch.no_grad():
            lm_loss, code_vec, nl_vec = model(code_inputs, nl_inputs)
            eval_loss += lm_loss.mean().item()
            code_vecs.append(code_vec.cpu().numpy())
            nl_vecs.append(nl_vec.cpu().numpy())
        nb_eval_steps += 1

    code_vecs = np.concatenate(code_vecs, 0)
    nl_vecs = np.concatenate(nl_vecs, 0)
    eval_loss = eval_loss / nb_eval_steps

    scores = np.matmul(nl_vecs, code_vecs.T)
    ranks = []
    for i in range(len(scores)):
        score = scores[i, i]
        rank = 1
        for j in range(len(scores)):
            if i != j and scores[i, j] >= score:
                rank += 1
        ranks.append(1 / rank)

    result = {
        "eval_loss": eval_loss,
        "eval_mrr": np.mean(ranks)
    }

    return result

class ContrastiveTrainerWithMRR(ContrastiveTrainer):
    def evaluate_model(self):
        # Call the evaluate function during evaluation
        eval_results = evaluate(training_args, model, tokenizer)
        print(f"Evaluation results - MRR: {eval_results['eval_mrr']}, Loss: {eval_results['eval_loss']}")

    def training_step(self, model, inputs):
        # Perform training step
        output = model(**inputs)
        
        # Save embeddings and evaluation
        step = self.state.global_step
        
        # Evaluate MRR at each epoch or interval (example: every 100 steps)
        if step % 100 == 0:
            self.evaluate_model()

        return super().training_step(model, inputs)

trainer = ContrastiveTrainerWithMRR(
    model,
    training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["val"],
    data_collator=lambda x: collate_fn(x, tokenizer),
)

# Start training with evaluation at each step or epoch
trainer.train()
