from transformers import RobertaTokenizer, RobertaModel
import json
from tqdm import tqdm
import torch
from torch.nn.functional import cosine_similarity

def get_model():
    model = RobertaModel.from_pretrained('microsoft/codebert-base')
    tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
    model = model.eval().cuda()

    return model, tokenizer

model, tokenizer = get_model()

import sys
from code_search import get_dataset, collate_fn, ContrastiveTrainer

languages = ["C", "PHP", "Java", "C++", "C#", "Javascript", "Python"]
root_path = "XLCoST_data"


dataset = get_dataset(root_path=root_path, languages=languages)

dataset

from transformers import TrainingArguments


training_args = TrainingArguments(
    "contrastive_trainer",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_steps=200,
    num_train_epochs=1,
    evaluation_strategy="no",
    report_to="none",
    remove_unused_columns=False,
    warmup_steps=1000,
    save_strategy="epoch"
)
trainer = ContrastiveTrainer(
    model,
    training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["val"],
    data_collator=lambda x: collate_fn(x, tokenizer),
)
# trainer.train()

def generate_predictions_jsonl(model, tokenizer, test_dataset, collate_fn, output_file="predictions.jsonl"):
    model.eval()
    predictions = []
    
    for item in tqdm(test_dataset, desc="Generating predictions"):
        batch = [item]
        inputs = collate_fn(batch, tokenizer)

        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]  # Assuming we take the [CLS] token
            # Compute similarity scores between query and code embeddings
            # similarity_scores = torch.matmul(outputs, outputs.T)  # Adjust similarity calculation as needed
            batch_scores = cosine_similarity(embeddings, embeddings)

            for i, score in enumerate(batch_scores):
                # Sort indices based on similarity scores
                sorted_indices = torch.argsort(score, descending=True).cuda().tolist()
                
                # Convert indices to prediction format, e.g., URLs or IDs
                predictions.append({
                    "data_idx": batch[i]["data_idx"],  # Assuming each item has a "url" key
                    "answers": [batch_scores[i].item()]
                })
    
    # Write to predictions.jsonl
    with open(output_file, "w") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")
    print(f"Predictions saved to {output_file}")

generate_predictions_jsonl(
    model=model,
    tokenizer=tokenizer,
    test_dataset=dataset["test"],  # assuming the test set is part of the dataset
    collate_fn=collate_fn,
    output_file="predictions.jsonl"
)
