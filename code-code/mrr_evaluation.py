import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel
from tqdm import tqdm
import numpy as np
import json
from peft import PeftModel, PeftConfig
from code_search import get_pooled_embeds

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
    inputs = tokenizer(" ".join(tokens), return_tensors="pt", padding="max_length", truncation=True)
    # print(type(inputs), inputs)
    # with torch.no_grad():
        # outputs = model(**inputs)
        # print(type(outputs), outputs)
    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    embeds = model(ids, attention_mask=mask)[0]
    print("ids: ", ids.shape, "mask: ", mask.shape, "embeds: ", embeds.shape)
    in_mask = mask.unsqueeze(-1).expand(embeds.size()).float()

    # careful here, we only want to pool embedds when it is NOT padding

    pooled_embeds = torch.sum(embeds * in_mask, 1) / torch.clamp(
            in_mask.sum(1), min=1e-6
    )
    print("pooled_embeds: ", pooled_embeds.shape)
    return pooled_embeds
    
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

def contrast_evaluation(query_embeds, code_embeds, ground_truth_indices):
    """
    Evaluates MRR and top-k accuracy (R@1, R@5, R@10) based on similarity scores.

    Parameters:
    - query_embeds (torch.Tensor): Query embeddings.
    - code_embeds (torch.Tensor): Relevant code embeddings.
    - ground_truth_indices (list of int): List of indices indicating correct matches.

    Returns:
    - eval_result (dict): Dictionary containing R@1, R@5, R@10, and MRR metrics.
    """
    score_matrix = query_embeds @ code_embeds.T
    scores = score_matrix.cpu().numpy()

    ranks = np.ones(scores.shape[0]) * -1
    for index, score in enumerate(scores):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == ground_truth_indices[index])[0][0]

    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    mrr = 100.0 * np.mean(1 / (ranks + 1))

    eval_result = {
        'r1': tr1,
        'r5': tr5,
        'r10': tr10,
        'mrr': mrr
    }
    return eval_result

def main_evaluation_script(file_path, model_name="microsoft/codebert-base", max_length=128):
    """
    Main evaluation script to compute R@1, R@5, R@10, and MRR for a code2code search task.
    
    Parameters:
    - file_path (str): Path to the test JSONL file.
    - model_name (str): Pre-trained model name for the tokenizer and model.
    - batch_size (int): Batch size for DataLoader.
    - max_length (int): Maximum sequence length for token embeddings.
    """
    data = load_data_from_jsonl(file_path)
    
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    print("Tokenizer max length: ", tokenizer.model_max_length)
    model = RobertaModel.from_pretrained(model_name)
    peft_model = PeftModel.from_pretrained(model, "schaturv/codebert-code2code-lora-r16", adapter_name="code2code")
    peft_model.eval()  # Set to evaluation mode
    peft_model.set_adapter("code2code")

    print(peft_model)

    print("Active adapters: ", peft_model.active_adapters)
    
    query_embeddings = []
    code_embeddings = []
    ground_truth_indices = []

    for idx, entry in enumerate(tqdm(data, desc="Generating embeddings")):
        query_embedding = compute_embeddings(model, tokenizer, entry["docstring_tokens"]).squeeze(0)
        code_embedding = compute_embeddings(model, tokenizer, entry["code_tokens"]).squeeze(0)

        query_embeddings.append(query_embedding)
        code_embeddings.append(code_embedding)
        ground_truth_indices.append(idx)  # In this case, the correct match is at the same index

    # Stack all embeddings for matrix operations
    query_embeddings = torch.stack(query_embeddings)
    code_embeddings = torch.stack(code_embeddings)

    eval_result = contrast_evaluation(query_embeddings, code_embeddings, ground_truth_indices)

    print(f"R@1: {eval_result['r1']:.2f}%")
    print(f"R@5: {eval_result['r5']:.2f}%")
    print(f"R@10: {eval_result['r10']:.2f}%")
    print(f"MRR: {eval_result['mrr']:.2f}%")

    return eval_result


if __name__ == "__main__":
    test_file_path = "../xlcost_data/retrieval/code2code_search/program_level/Python/test.jsonl"
    model_name = "microsoft/codebert-base"
    avg_mrr = main_evaluation_script(file_path=test_file_path, model_name=model_name)
