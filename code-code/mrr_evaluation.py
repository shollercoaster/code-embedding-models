import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel
from tqdm import tqdm
import numpy as np
import json

class CodeEmbeddingDataset(Dataset):
    def __init__(self, data):
        """
        Initialize dataset with structured data containing docstring_tokens (query) and code_tokens (relevant code).
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns a single data item consisting of query tokens and relevant code tokens.
        """
        return {
            "query_tokens": self.data[idx]["docstring_tokens"],
            "relevant_tokens": self.data[idx]["code_tokens"],
            "idx": self.data[idx]["idx"]
        }

def create_dataset(data):
    """
    Convert raw data into a list of dictionaries compatible with CodeEmbeddingDataset.
    """
    dataset = [{"docstring_tokens": item["docstring_tokens"],
                "code_tokens": item["code_tokens"],
                "idx": item["idx"]} for item in data]
    return CodeEmbeddingDataset(dataset)

def compute_embeddings(model, tokenizer, tokens):
    """
    Convert tokens into embeddings using a code embedding model.
    """
    inputs = tokenizer(tokens, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

def evaluate_mrr(model, tokenizer, dataset):
    """
    Evaluates the Mean Reciprocal Rank (MRR) for code2code search task.
    """
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    mrr_scores = []

    for batch in tqdm(dataloader, desc="Evaluating MRR"):
        query_tokens = batch["query_tokens"][0]
        relevant_tokens = batch["relevant_tokens"][0]

        query_embedding = compute_embeddings(model, tokenizer, query_tokens).squeeze(0)

        relevant_embeddings = compute_embeddings(model, tokenizer, relevant_tokens)

        similarities = torch.cosine_similarity(query_embedding, relevant_embeddings)

        ranked_indices = torch.argsort(similarities, descending=True)
        rank = (ranked_indices == 0).nonzero(as_tuple=True)[0].item() + 1  # Position of relevant item (index 0)

        mrr_scores.append(1 / rank)

    avg_mrr = np.mean(mrr_scores)
    print(f"Average MRR: {avg_mrr:.4f}")
    return avg_mrr

def load_data_from_jsonl(file_path):
    """
    Loads data from a JSONL file.

    Parameters:
    - file_path (str): Path to the JSONL file.

    Returns:
    - data (list of dict): List of data points with fields `docstring_tokens`, `code_tokens`, and `idx`.
    """
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            record = json.loads(line.strip())
            data.append({
                "idx": record["idx"],
                "docstring_tokens": record["docstring_tokens"],
                "code_tokens": record["code_tokens"]
            })
    return data

def main_evaluation_script(file_path, model_name="microsoft/codebert-base", batch_size=1):
    """
    Main evaluation script to compute the average MRR for a code2code search task.
    
    Parameters:
    - file_path (str): Path to the test JSONL file.
    - model_name (str): Pre-trained model name for the tokenizer and model.
    - batch_size (int): Batch size for DataLoader.
    """
    data = load_data_from_jsonl(file_path)
    
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaModel.from_pretrained(model_name)
    model.eval()  # Set to evaluation mode
    
    dataset = create_dataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    mrr_scores = []
    
    for batch in tqdm(dataloader, desc="Evaluating MRR"):
        query_tokens = batch["query_tokens"][0]
        relevant_tokens = batch["relevant_tokens"][0]

        query_embedding = compute_embeddings(model, tokenizer, query_tokens).squeeze(0)

        relevant_embeddings = compute_embeddings(model, tokenizer, relevant_tokens)

        similarities = torch.cosine_similarity(query_embedding, relevant_embeddings)

        ranked_indices = torch.argsort(similarities, descending=True)

        rank = (ranked_indices == 0).nonzero(as_tuple=True)[0].item() + 1

        mrr_scores.append(1 / rank)

    avg_mrr = np.mean(mrr_scores)
    print(f"Final Average MRR: {avg_mrr:.4f}")

    return avg_mrr


if __name__ == "__main__":
    test_file_path = "../xlcost_data/retrieval/code2code_search/program_level/C/test.jsonl"
    model_name = "microsoft/codebert-base"
    avg_mrr = main_evaluation_script(file_path=test_file_path, model_name=model_name)
