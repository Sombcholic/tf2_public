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

model_final = load_model('./save_model/20230725-V1-hospital_bert_model_epoch20.h5', custom_objects={'KerasLayer':hub.KerasLayer})

# 利用opencc 簡轉中
cc = OpenCC('s2twp')



sentence = [['心臟有點痛']]

for data in sentence:
    result = model_final.predict(data)
    print(result)