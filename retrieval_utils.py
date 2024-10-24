import os
import json
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, Any, Tuple, Optional, Set
import torch
import numpy as np
from typing import Dict
import torch.nn.functional as F


def get_rank_of_key(
    query_embedding: torch.Tensor,
    database_embeddings: torch.Tensor,
    query_idx: str,
    database_idxs: np.array,
    key_index: int,
) -> torch.Tensor:
    """
    Given a query embedding and a tensor of database embeddings, this function calculates
    the cosine similarity between the query and each database item, sorts them in descending
    order, and then returns the rank (1-indexed) of a specified key (represented by 'key_index')
    in the sorted list. It sets the score to -10 at all indexes where query_idx equals database_idxs,
    except for the first one.

    Parameters:
    -----------
    query_embedding: torch.Tensor
        The query embedding tensor. Should be 1D tensor of shape (d,), where 'd' is the
        dimensionality of the embeddings.

    database_embeddings: torch.Tensor
        The database embeddings tensor. Should be a 2D tensor of shape (n, d), where 'n' is
        the number of items in the database, and 'd' is the dimensionality of the embeddings.

    query_idx: str
        The index of the query in the database.

    database_idxs: np.array
        A numpy array containing the indices corresponding to the embeddings in the database.
        Should be a 1D array of shape (n,), where 'n' is the number of items in the database.

    key_index: int
        The index of the key in the database embeddings tensor for which we want to find the
        rank.

    Returns:
    --------
    torch.Tensor
        A tensor containing the rank (1-indexed) of the specified key in the sorted list of scores.
    """

    scores = F.cosine_similarity(query_embedding[None, :], database_embeddings)
    assert scores.shape == (len(database_embeddings),)
    first_match_pos = (database_idxs == query_idx).nonzero()[0][0]
    assert first_match_pos == key_index

    mask = (database_idxs == query_idx)[first_match_pos + 1 :]
    scores[first_match_pos + 1 :][mask] = -100
    _, sorted_indices = torch.sort(scores, descending=True)
    key_rank = (sorted_indices == key_index).nonzero(as_tuple=True)[0] + 1
    return key_rank


def ranking_metrics(ranks: torch.Tensor) -> Dict[str, float]:
    """
    Given a tensor of ranks, this function calculates the Mean Reciprocal Rank (MRR)
    and Recall@K for K={1,3,5}.

    Parameters:
    -----------
    ranks: torch.Tensor
        The tensor of ranks. Should be 1D tensor of shape (n,), where 'n' is the number of
        items for which we have ranks.

    Returns:
    --------
    Dict[str, float]
        A dictionary with keys 'mrr', 'recall@1', 'recall@3', and 'recall@5'. The values are
        the respective metric scores.
    """

    mrr_scores = 1 / ranks.float()

    recall_at_1_scores = (ranks <= 1).float()
    recall_at_3_scores = (ranks <= 3).float()
    recall_at_5_scores = (ranks <= 5).float()

    ranking_metrics = {
        "mrr": mrr_scores.mean().item(),
        "recall@1": recall_at_1_scores.mean().item(),
        "recall@3": recall_at_3_scores.mean().item(),
        "recall@5": recall_at_5_scores.mean().item(),
    }

    return ranking_metrics


def construct_database_and_perform_search(
    source_language: str,
    target_language: str,
    other_languages: Set[str],
    data_dict: Dict[str, Dict[str, Dict[str, Any]]],
    embedding_field: str,
    save_dir: str,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Construct a database from given languages and perform a search for items from a source language.

    This function calculates the cosine similarity between each item of the source language
    and the items in a database consisting of items from the target language and optionally
    other languages. It returns the Mean Reciprocal Rank (MRR), and Recall@k for k=1,3,5
    for the ranks of the source items in the sorted list of cosine similarities.

    The database can be monolingual (only target language) or multilingual
    (target language plus other languages), based on whether other_languages is an empty set or not.

    Parameters:
    -----------
    source_language: str
        The source language.

    target_language: str
        The target language.

    other_languages: Set[str]
        A set of other languages. If this set is empty, the database will be monolingual.
        Otherwise, it will be multilingual.

    data_dict: Dict[str, Dict[str, Dict[str, Any]]]
        The data dictionary. Keys are languages and values are dictionaries where the key is an
        id and the value is a dictionary containing the 'embedding' tensor and other data related
        to the item.

    embedding_field: str
        The field in the data dictionary where embeddings are stored. This specifies the
        location in each data entry where the embedding can be found, and is used to pull out
        the embeddings for comparison.

    save_dir: str
        The directory where to save the metrics.

    verbose: bool, default=False
        If True, print additional information during processing.

    Returns:
    --------
    Dict[str, float]
        A dictionary containing the "mrr", "recall@1", "recall@3", and "recall@5" metrics.
    """

    file_path = os.path.join(
        save_dir, f"{embedding_field}_{source_language}_{target_language}.json"
    )

    if os.path.exists(file_path):
        print(f"The metrics for {embedding_field} and {source_language} already exist at: {file_path}.")
        return None 


    lang_idx_embedding_dict = {}
    for lang, data in data_dict.items():
        idxs = [key for key in data.keys()]
        embeddings = torch.stack([value[embedding_field] for value in data.values()])
        lang_idx_embedding_dict[lang] = {"idxs": idxs, embedding_field: embeddings}

    source_dict = lang_idx_embedding_dict[source_language]
    target_dict = lang_idx_embedding_dict[target_language]

    query_embeddings = source_dict[embedding_field]
    database_idxs = np.array(
        target_dict["idxs"]
        + [
            idx
            for lang in other_languages
            for idx in lang_idx_embedding_dict[lang]["idxs"]
        ],
        dtype=object,
    )
    database_embeddings = torch.cat(
        [target_dict[embedding_field]]
        + [lang_idx_embedding_dict[lang][embedding_field] for lang in other_languages],
        dim=0,
    )

    ranks_list = []
    for pos_query_idx, query_idx in tqdm(enumerate(source_dict["idxs"])):
        if query_idx not in target_dict["idxs"]:
            continue
        key_index = target_dict["idxs"].index(query_idx)
        query_embedding = query_embeddings[pos_query_idx]
        rank = get_rank_of_key(
            query_embedding=query_embedding,
            database_embeddings=database_embeddings,
            query_idx=query_idx,
            database_idxs=database_idxs,
            key_index=key_index,
        )
        ranks_list.append(rank)

    ranks = torch.stack(ranks_list)
    metrics = ranking_metrics(ranks=ranks)
    if verbose:
        print(f"Source language: {source_language}")
        print(f"Target language: {target_language}")
        print(f"Embedding field: {embedding_field}")
        print(f"Number of queries made: {len(ranks)}")
        print(f"Database size: {len(database_idxs)}")
        print(f"Metrics: {metrics}")
        print(f"Metrics are saved in the directory: {save_dir}")
   
    with open(file_path, "w") as f:
        json.dump(metrics, f)

    return metrics

def calculate_mrr_at_step(model, query_embeddings, database_embeddings, query_idxs, database_idxs):
    """
    Calculates MRR during each finetuning step.
    
    Parameters:
    -----------
    model: torch.nn.Module
        The finetuned model for generating embeddings.
    
    query_embeddings: torch.Tensor
        Embeddings for the query (current batch).
    
    database_embeddings: torch.Tensor
        The embeddings from the database to compare against.
    
    query_idxs: list
        A list of indices representing the queries.
    
    database_idxs: list
        A list of indices representing the database entries.
    
    Returns:
    --------
    float:
        The calculated MRR score for the current step.
    """
    ranks_list = []
    
    for i, query_embedding in enumerate(query_embeddings):
        # Retrieve the corresponding query index
        query_idx = query_idxs[i]
        if query_idx not in database_idxs:
            continue
        
        key_index = database_idxs.index(query_idx)
        rank = get_rank_of_key(
            query_embedding=query_embedding,
            database_embeddings=database_embeddings,
            query_idx=query_idx,
            database_idxs=np.array(database_idxs),
            key_index=key_index
        )
        ranks_list.append(rank)
    
    ranks = torch.stack(ranks_list)
    metrics = ranking_metrics(ranks)
    
    return metrics['mrr']  # Return only MRR for this step

def finetune_with_mrr(model, train_dataloader, optimizer, database_embeddings, database_idxs, num_epochs):
    """
    Finetunes the model and calculates MRR at each step during training.
    
    Parameters:
    -----------
    model: torch.nn.Module
        The model being finetuned.
    
    train_dataloader: torch.utils.data.DataLoader
        The DataLoader for the training data.
    
    optimizer: torch.optim.Optimizer
        The optimizer for the training process.
    
    database_embeddings: torch.Tensor
        The embeddings of the database.
    
    database_idxs: list
        The indices corresponding to the database entries.
    
    num_epochs: int
        Number of epochs to finetune the model.
    
    """
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            # Assuming batch contains query inputs and labels for the loss calculation
            query_inputs, labels = batch
            optimizer.zero_grad()
            
            # Forward pass
            query_embeddings = model(query_inputs)
            
            # Compute the loss (for example, using CrossEntropyLoss, depending on the task)
            loss = F.cross_entropy(query_embeddings, labels)
            loss.backward()
            optimizer.step()
            
            # Update total loss
            epoch_loss += loss.item()
            
            # Calculate MRR at each step
            query_idxs = labels.tolist()  # Assuming labels correspond to the query indices
            mrr_at_step = calculate_mrr_at_step(
                model, query_embeddings, database_embeddings, query_idxs, database_idxs
            )
            
            # Log the loss and MRR for the step
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}, MRR: {mrr_at_step:.4f}")
        
        # Print the epoch loss
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {epoch_loss / len(train_dataloader):.4f}")
