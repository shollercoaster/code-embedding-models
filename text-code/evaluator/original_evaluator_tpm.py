import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
from data_utils import create_dataset, create_loader
from peft import PeftModel, PeftConfig

from torch import Tensor

@torch.no_grad()
def get_feats(model, tokenizer, data_loader, max_length, device, desc='Get feats'):
    embeds = []

    for text in tqdm(data_loader, total=len(data_loader), desc=desc):
        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=max_length,
                               return_tensors="pt").to(device)
        ids = text_input["input_ids"]
        mask = text_input["attention_mask"]
        embed = model(ids, attention_mask=mask)[0]
        in_mask = mask.unsqueeze(-1).expand(embed.size()).float()
        pooled_embeds = torch.sum(embed * in_mask, 1) / torch.clamp(
                in_mask.sum(1), min=1e-6
        )
        embeds.append(pooled_embeds)

    # embeds = torch.cat(embeds, dim=0)

    return embeds


@torch.no_grad()
def contrast_evaluation(text_embeds, code_embeds, img2txt):
    score_matrix_i2t = text_embeds @ code_embeds.t()
    scores_i2t = score_matrix_i2t.cpu().numpy()


    ranks = np.ones(scores_i2t.shape[0]) * -1
    for index, score in enumerate(scores_i2t):
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
_, _, test_dataset, code_dataset = create_dataset('../../dataset/CSN', 'python')

test_loader, code_loader = create_loader([test_dataset, code_dataset], [None, None],
                                             batch_size=[256, 256],
                                             num_workers=[4, 4], is_trains=[False, False], collate_fns=[None, None])

tokenizer = AutoTokenizer.from_pretrained('Salesforce/codet5p-110m-embedding', trust_remote_code=True)
model = AutoModel.from_pretrained('Salesforce/codet5p-110m-embedding', trust_remote_code=True)

peft_model = PeftModel.from_pretrained(model, "schaturv/graphcodebert-text2code-lora-r16", adapter_name="text2code")
peft_model.eval()  # Set to evaluation mode
peft_model.set_adapter("text2code")

print(peft_model)

print("Active adapters: ", peft_model.active_adapters)

print('\nStart zero-shot evaluation...')
device = torch.device('cuda')
model = peft_model.to(device)
model.eval()
print("Active adapters: ", peft_model.active_adapters)

text_embeds = get_feats(model, tokenizer, test_loader, 64, device, desc='Get text feats')
code_embeds = get_feats(model, tokenizer, code_loader, 360, device, desc='Get code feats')
test_result = contrast_evaluation(text_embeds, code_embeds, test_loader.dataset.text2code)
print(f'\n====> zero-shot test result: ', test_result)

