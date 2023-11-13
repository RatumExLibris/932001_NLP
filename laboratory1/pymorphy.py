import re
from tqdm import tqdm

import nltk
from nltk.tokenize import word_tokenize
import pymorphy2

nltk.download('punkt')

file = open('Capital_Karl_Marx.txt', 'r')
text = file.read()

words = word_tokenize(text)

morphy = pymorphy2.MorphAnalyzer()
result = []
left_word = { 'word': '', 'POS': '', 'gender': '', 'number': '', 'case': '', 'idx':-2 }

for i in tqdm(range(0, len(words))):
    m = morphy.parse(words[i])[0]
    tag = m.tag
    if tag.POS in ['NOUN', 'ADJF']:
        if (i-left_word['idx']==1 and left_word['gender']==tag.gender and
            left_word['number']==tag.number and left_word['case']==tag.case and
            left_word['POS']!=tag.POS):
            result.append(left_word['word'] + ' ' + m.normal_form)
        left_word['idx'] = i
        left_word['word'] = m.normal_form
        left_word['POS'] = tag.POS
        left_word['gender'] = tag.gender
        left_word['number'] = tag.number
        left_word['case'] = tag.case

result1 = list(set(result))

print(result1[0:10])

f = open("task1_results.txt", "a")
f.write('\n'.join(result1))
f.close()
