#!/usr/bin/env python
# coding: utf-8

# In[8]:


import re
from tqdm import tqdm
from rnnmorph.predictor import RNNMorphPredictor
import nltk
from nltk.tokenize import word_tokenize
import pandas as pd

# Выполнение этого пункта займёт много времени

nltk.download('punkt')

file = open('Capital_Karl_Marx.txt', 'r')
text = file.read()

words = word_tokenize(text)

predictor = RNNMorphPredictor(language='ru')

res = predictor.predict(words)

nf = []
word = []
pos = []
tag = []
score = []
for i in res:
    nf.append(i.normal_form)
    word.append(i.word)
    pos.append(i.pos)
    tag.append(i.tag)
    score.append(i.score)
df = pd.DataFrame({'normal_form': nf, 'word': word, 'pos': pos, 'tag':tag, 'score':score})
df.head()


df.to_csv('RNNwords.csv', index=False)


# <h3>Поиск пар слов по предоставленным условиям</h3>

df = pd.read_csv('RNNwords.csv')

def get_tags_from_str(tag_str):
    params = tag_str.split('|')
    tags = {}
    if len(params)>1:
        for i in params:
            tag = i.split('=')
            tags[tag[0]] = tag[1]
    return tags

result = []
left_word = { 'word': '', 'POS': '', 'gender': '', 'number': '', 'case': '', 'idx':-2 }
idxs = df.index.to_list()

for i in tqdm(range(0, df.shape[0])):
    tag = get_tags_from_str(df.iloc[i].tag)
    if df.iloc[i].pos in ['NOUN', 'ADJ'] and len(set(tag.keys()).intersection(set(['Gender', 'Case', 'Number'])))==3:
        if (i-left_word['idx']==1 and left_word['gender']==tag['Gender'] and
            left_word['number']==tag['Number'] and left_word['case']==tag['Case'] and
            left_word['POS']!=df.iloc[i].pos):
            result.append(left_word['word'] + ' ' + df.iloc[i].normal_form)
        left_word['idx'] = i
        left_word['word'] = df.iloc[i].normal_form
        left_word['POS'] = df.iloc[i].pos
        left_word['gender'] = tag['Gender']
        left_word['number'] = tag['Number']
        left_word['case'] = tag['Case']

# Drop duplicates
result1 = list(set(result))

file = open('task1_results.txt', 'r')
task1_result = file.read().split('\n')
print(task1_result[0:30])

res1_task1_diff = list(set(result1).difference(set(task1_result)))
res1_task1_diff.sort()
print(res1_task1_diff[0:30])

# <h3>Свойства RNN</h3>
# 
# - больше правильно определённых тегов на основе контекста
# - понимаются сокращения, слова по смыслу схожие с прилагательными
# - числительные принимаются за прилагательные
# - понимает зашумлённые слова. Например: ру-ка, часто из-за этого возникают ошибки

task1_res1_diff = list(set(task1_result).difference(set(result1)))
task1_res1_diff.sort()
print(task1_res1_diff[0:30])

# <h3>Свойства PyMorphy</h3>
# 
# - местоимения выдаются за прилагательные

intersection = list(set(result1).intersection(set(task1_result)))
intersection.sort()
print(intersection[0:30])

# <h3>Объединение</h3>
# 
# - При объединении результаты получаются чище
