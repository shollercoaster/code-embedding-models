# -*- coding: utf-8 -*-
"""openai_embeddings.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/github/MeharSinghShienh/code-search-benchmarking/blob/main/code_search_models/openai_embeddings.ipynb
"""

import numpy as np
from tqdm import tqdm
from data_utils import create_dataset, create_loader

import openai
#enter openai api key
openai.api_key = ("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")

from openai.embeddings_utils import get_embedding, cosine_similarity

def get_feats(data_loader, desc='Get feats'):
    embeds = []
    max_token_length = 29120

    for text in tqdm(data_loader, total=len(data_loader), desc=desc):

        for txt in text:
            txt = txt[0:max_token_length]
            embed = get_embedding(txt, engine='text-embedding-ada-002')
            embeds.append(embed)

    return embeds


def contrast_evaluation(text_embeds, code_embeds, img2txt):
    text_embeds = np.array(text_embeds)
    code_embeds = np.array(code_embeds)

    # Initialize the score matrix
    score_matrix_i2t = np.zeros((text_embeds.shape[0], code_embeds.shape[0]))

    # Compute cosine similarity for each pair of embeddings
    for i, text_embed in enumerate(text_embeds):
        for j, code_embed in enumerate(code_embeds):
            score_matrix_i2t[i, j] = cosine_similarity(text_embed, code_embed)

    ranks = np.ones(score_matrix_i2t.shape[0]) * -1

    for index, score in enumerate(score_matrix_i2t):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == img2txt[index])[0][0]


    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    mrr = 100.0 * np.mean(1 / (ranks + 1))


    eval_result = {'r1': tr1,
                   'r5': tr5,
                   'r10': tr10,
                   'mrr': mrr}
    return eval_result

print("\nCreating retrieval dataset")
#change language and path to dataset here
_, _, test_dataset, code_dataset = create_dataset('dataset/CSN', 'ruby')

test_loader, code_loader = create_loader([test_dataset, code_dataset], [None, None],
                                         batch_size=[128, 128],
                                         num_workers=[4, 4], is_trains=[False, False], collate_fns=[None, None])


print('\nStart zero-shot evaluation...')

text_embeds = get_feats(test_loader, desc='Get text feats')
code_embeds = get_feats(code_loader, desc='Get code feats')
test_result = contrast_evaluation(text_embeds, code_embeds, test_loader.dataset.text2code)

print(f'\n====> zero-shot test result: ', test_result)

