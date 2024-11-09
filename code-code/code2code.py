import os
import sys
import json
from tqdm import tqdm
import torch
from code_search import get_dataset, collate_fn, collate_fn_concatenated, ContrastiveTrainer
from transformers import RobertaTokenizer, RobertaModel, TrainingArguments
from torch.nn.functional import cosine_similarity

def get_model():
    model = RobertaModel.from_pretrained('microsoft/codebert-base')
    tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
    model = model.eval().cuda()

    return model, tokenizer

model, tokenizer = get_model()

languages = ["C", "PHP", "Java", "C++", "C#", "Javascript", "Python"]
root_path = "xlcost_data"

dataset = get_dataset(root_path=root_path, languages=languages)

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
    save_strategy="epoch",
    dataloader_pin_memory=False
)
trainer = ContrastiveTrainer(
    model,
    training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["val"],
    data_collator=lambda x: collate_fn(x, tokenizer),
)

# trainer.train()

def generate_predictions_jsonl(model, tokenizer, test_dataset, output_file="predictions.jsonl"):
    """
    Generates predictions for each query in the test dataset, outputting similarity scores.

    Args:
        model (PreTrainedModel): The fine-tuned model.
        tokenizer (PreTrainedTokenizer): The tokenizer associated with the model.
        test_dataset (Dataset): The formatted test dataset with `query` and `relevant_code` fields.
        output_file_path (str): Path where the predictions JSONL will be saved.
    """
    model.eval()
    
    predictions = []
    
    for item in tqdm(test_dataset, desc="Generating predictions"):
        batch = [item]
        inputs = collate_fn_concatenated(batch, tokenizer)

        with torch.no_grad():
            outputs = model(**inputs)
            batch_scores = cosine_similarity(outputs.last_hidden_state[:, 0, :], outputs.last_hidden_state[:, 1, :])

            for i, item in enumerate(batch):                
                predictions.append({
                    "data_idx": batch[i]["data_idx"],
                    "answers": [batch_scores[i].item()]
                })
    
    # Write to predictions.jsonl
    with open(output_file, "w") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")
    print(f"Predictions saved to {output_file}")

base_path = 'xlcost_data/retrieval/code2code_search'
levels = ["program_level", "snippet_level"]
# languages = ["C", "C++", "C#", "Java", "Javascript", "PHP", "Python"]

for level in levels:
    for language in languages:
        test_dataset = dataset["test"]
        output_file = f"{base_path}/{level}/{language}/predictions.jsonl"
        generate_predictions_jsonl(model, tokenizer, test_dataset, output_file)
