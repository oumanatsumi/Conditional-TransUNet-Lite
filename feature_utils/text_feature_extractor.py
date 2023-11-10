from transformers import BertTokenizer, BertModel
import torch

"""
    返回一个torch.Size([1, 768])
"""
def text_embedding(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    model = BertModel.from_pretrained('bert-base-chinese')
    inputs = tokenizer(str(text), return_tensors='pt')
    outputs = model(**inputs)
    pooler_output = outputs.pooler_output
    return pooler_output
    # print(pooler_output.shape)

