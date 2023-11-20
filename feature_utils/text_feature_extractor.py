from transformers import BertTokenizer, BertModel
import torch
import jieba
import re
from fuzzywuzzy import fuzz
from difflib import SequenceMatcher
from fuzzywuzzy import process


"""
    返回一个torch.Size([1, 768])
"""
def text_embedding(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    model = BertModel.from_pretrained('bert-base-chinese')
    inputs = tokenizer(str(text), return_tensors='pt')
    outputs = model(**inputs)
    hidden_states = outputs.last_hidden_state
    return hidden_states
    # print(pooler_output.shape)

text = ("经CT检查，病人主动脉瓣区域钙化程度较高且分布广泛。病人主动脉瓣瓣叶对称性良好")


# str = re.sub('[^\w]', '', text)  # 使用正则去符号，之后都是用这个str字符串
seg_list = jieba.lcut(text, cut_all=False)  # 精确模式
print('  '.join(seg_list))
key = "钙化区域钙化程度"
word = process.extract("钙化区域钙化程度", seg_list, limit= 3)
# times = seg_list.count(word[0])
print(word)
# print(times)
res = ""
for seg in seg_list:
    res = res + "  "+str(fuzz.ratio(seg, key))
print(res)
hidden = text_embedding(text)
print(hidden)