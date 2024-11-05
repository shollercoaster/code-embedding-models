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
import torch.nn.functional as F

class CustomRobertaModel(RobertaModel):
    def forward(self, input_ids_query=None, attention_mask_query=None, input_ids_relevant=None, attention_mask_relevant=None, **kwargs):
        # Process query inputs
        query_outputs = None
        if input_ids_query is not None:
            query_outputs = self.roberta(
                input_ids=input_ids_query,
                attention_mask=attention_mask_query,
                **kwargs
            )
        
        # Process relevant inputs
        relevant_outputs = None
        if input_ids_relevant is not None:
            relevant_outputs = self.roberta(
                input_ids=input_ids_relevant,
                attention_mask=attention_mask_relevant,
                **kwargs
            )
        
        return query_outputs, relevant_outputs


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
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Multi-GPU support
    # if args.n_gpu > 1 and eval_when_training is False:
    #    model = torch.nn.DataParallel(model)

    print("***** Running evaluation *****")
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    code_vecs_query = []
    code_vecs_relevant = []
    
    for batch in eval_dataloader:
        query_code_tokens = tokenizer(batch['query_code'], return_tensors='pt', padding=True, truncation=True, max_length=512)
        relevant_code_tokens = tokenizer(batch['relevant_code'], return_tensors='pt', padding=True, truncation=True, max_length=512)

        code_inputs_query = query_code_tokens['input_ids'].to(args.device)
        query_attention_mask = query_code_tokens['attention_mask'].to(args.device)

        code_inputs_relevant = relevant_code_tokens['input_ids'].to(args.device)
        relevant_attention_mask = relevant_code_tokens['attention_mask'].to(args.device)


        with torch.no_grad():
            lm_loss, code_vec_query = model(
                input_ids=code_inputs_query, 
                attention_mask=query_attention_mask,
                # input_ids_relevant=code_inputs_relevant,
                # attention_mask_relevant=relevant_attention_mask
            )

            eval_loss += lm_loss.mean().item()
            code_vecs_query.append(code_vec_query.cpu().numpy())
            code_vecs_relevant.append(code_vec_relevant.cpu().numpy())
        
        nb_eval_steps += 1

    code_vecs_query = np.concatenate(code_vecs_query, 0)
    code_vecs_relevant = np.concatenate(code_vecs_relevant, 0)
    eval_loss = eval_loss / nb_eval_steps

    # Calculate MRR
    mrr = calculate_mrr(code_vecs_query, code_vecs_relevant)

    result = {
        "eval_loss": eval_loss,
        "eval_mrr": mrr
    }

    return result

def calculate_mrr(code_vecs_query, code_vecs_relevant):
    ranks = []
    for i in range(len(code_vecs_query)):
        score = F.cosine_similarity(torch.tensor(code_vecs_query[i]), torch.tensor(code_vecs_relevant))
        score[i] = -100  # Exclude the query itself from scoring
        sorted_indices = torch.argsort(score, descending=True)
        rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(1 / rank)

    return np.mean(ranks)

class ContrastiveTrainerWithMRR(ContrastiveTrainer):
    def evaluate_model(self):
        # Call the evaluate function during evaluation
        eval_results = evaluate(training_args, model, tokenizer)
        print(f"Evaluation results - MRR: {eval_results['eval_mrr']}, Loss: {eval_results['eval_loss']}")

    def training_step(self, model, inputs):
        query_input_ids = inputs['query']['input_ids']
        query_attention_mask = inputs['query']['attention_mask']
        
        relevant_input_ids = inputs['relevant']['input_ids']
        relevant_attention_mask = inputs['relevant']['attention_mask']
        
        # Prepare the inputs for the model
        query_inputs = {
            'input_ids': query_input_ids,
            'attention_mask': query_attention_mask,
        }
        relevant_inputs = {
            'input_ids': relevant_input_ids,
            'attention_mask': relevant_attention_mask,
        }
        labels = inputs['labels']  # Ensure labels are also on the right device
        
        query_output = model(**query_inputs)
        #relevant_outputs = model(**relevant_inputs)
        
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
