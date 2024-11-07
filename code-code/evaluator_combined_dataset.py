import os
import logging
import sys
import json
import numpy as np
from typing import Dict, List, Any
from datasets import DatasetDict
from code_search import get_dataset

def read_answers(data: DatasetDict) -> Dict[str, str]:
    answers = {}
    for item in data["test"]:
        answers[item["data_idx"]] = f"{item['src_id']}/{item['trgt_id']}"
    return answers

def read_predictions(filename: str) -> Dict[str, List[str]]:
    predictions = {}
    with open(filename) as f:
        for line in f:
            js = json.loads(line.strip())
            predictions[js["data_idx"]] = js["answers"]
    return predictions

def calculate_scores(answers: Dict[str, str], predictions: Dict[str, List[str]]) -> Dict[str, float]:
    scores = []
    for key in answers:
        if key not in predictions:
            logging.error(f"Missing prediction for data_idx {key}.")
            sys.exit()
        query_scores = []
        for rank, idx in enumerate(predictions[key]):
            if idx.split("/")[0] == answers[key].split("/")[0]:
                query_scores.append(1 / (rank + 1))
        scores.append(np.mean(query_scores) if query_scores else 0)
    return {"MRR": round(np.mean(scores), 4)}

def precision_at_k(answers: Dict[str, str], predictions: Dict[str, List[str]], k: int = 6) -> Dict[str, float]:
    scores = []
    for key in answers:
        if key not in predictions:
            logging.error(f"Missing prediction for data_idx {key}.")
            sys.exit()
        precision_count = sum(1 for idx in predictions[key][:k] if idx.split("/")[0] == answers[key].split("/")[0])
        scores.append(precision_count / k)
    return {f"Precision@{k}": round(np.mean(scores), 4)}

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate cumulative predictions on the combined test dataset.")
    parser.add_argument("--root_data_path", "-d", help="Root path of the dataset.", required=True)
    parser.add_argument("--predictions", "-p", help="Filename of the cumulative predictions in JSONL format.", required=True)
    
    args = parser.parse_args()
    languages = ["C", "C#", "C++", "Java", "Javascript", "PHP", "Python"]
    combined_data = get_dataset(args.root_data_path, languages)
    
    answers = read_answers(combined_data)
    predictions = read_predictions(args.predictions)
    
    scores = calculate_scores(answers, predictions)
    precision = precision_at_k(answers, predictions)
    
    print(scores)
    print(precision)

if __name__ == "__main__":
    main()
