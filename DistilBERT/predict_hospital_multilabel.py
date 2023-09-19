# https://www.kaggle.com/code/nayansakhiya/text-classification-using-bert/notebook
# import tokenization
from bert.tokenization import bert_tokenization
import tensorflow as tf
import tensorflow_hub as hub
from keras.utils import to_categorical
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from keras.models import load_model
import tensorflow_text as text
import yaml
import copy
from opencc import OpenCC

model_final = load_model('./save_model/20230727-V1-hospital_bert_model_epoch50.h5', custom_objects={'KerasLayer':hub.KerasLayer})

# 利用opencc 簡轉中
cc = OpenCC('s2twp')

with open('./data/20230725_V1_hospitallabelList.yml', 'r', encoding='utf-8') as f:
    labelList = yaml.safe_load(f)

with open('./data/20230725_V1_hospitalsickList.yml', 'r', encoding='utf-8') as f:
    sickList = yaml.safe_load(f)

sentence = [
    ['我的頸部腫脹，有種緊繃的感覺'], 
    ['我的食慾減退，體重下降。'], 
    ['我的排尿習慣有所改變，例如頻繁或稀少。'],
    ['我發現乳房上有凸起物']
]

answer = []

for data in sentence:
    result = model_final.predict(data)
    _tmp = []
    _sick = []

    # print('see see')
    # print(result[0])

    for probability in result[0]:
        
        i = 0
        for p in probability:
            if (p >= 0.60):
                _tmp.append(labelList[i])
            i += 1

    # print('第二個')
    # print()
    # print(sickList[np.argmax(result[1], axis=1)[0]])

    for probability in result[1]:
        i = 0
        for p in probability:
            if (p >= 0.60):
                _sick.append(sickList[i])
            i += 1

    if (_sick == []):
        _sick = [sickList[np.argmax(result[1], axis=1)[0]]]

    answer.append([_tmp, _sick])

    # print(answer)

    i = 0
    for a in answer:
        print('使用者問得語句：')
        print(sentence[i][0])
        print(a)
        print()
        i += 1
    # print(result)