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

# Build a BERT layer
# bert_url = './model/distilbert_multi_cased_L-6_H-768_A-12_1'
# bert_layer = hub.KerasLayer(bert_url, trainable=True)

# preprocess_url  = './model/distilbert_multi_cased_preprocess_2'
# preprocessor = hub.KerasLayer(preprocess_url)

input_data = [
    ['我希望查詢電信公司'],
    ['我想知道菲律賓'],
    ['我打算'],
    ['我想要檢視我在'],
    ['我想查看看我的G'],
    ['希望查詢電信公司的']
]

keyword_index = None

user_input = []

for data in input_data:
    user_input.append(data[0].lower())


df = pd.read_csv('./data/keyword/20230620-Test1-class16_generate_dict.csv', encoding='utf-8')


# print(df.iloc[1065]['word'])

model_final = load_model('./save_model/20230719-V1-generate_bert_model_epoch_tfdataset5.h5', custom_objects={'KerasLayer':hub.KerasLayer})

# y_predict = model_final.predict(input_data[0])

# result = np.argmax(y_predict, axis=1)

# print('lets see')
# print(np.argsort(y_predict)[0][-2])

# np.argsort(np.max(x, axis=0))[-2]

# print(result)

# for i in range(10):
#     num = np.random.randint(2)
#     print('see see')
#     print(num)

# sys.exit(0)

generate_sentence = []
different_ans = False


# 利用opencc 簡轉中
cc = OpenCC('s2twp')


for data in user_input:
    sentence = copy.deepcopy(data)
    result = [0]

    while (result[0] != 11):
        y_predict = model_final.predict([sentence])

        if (different_ans is False):
            result = np.argmax(y_predict, axis=1)
        else:
            num = np.random.randint(2)
            

            if (np.argsort(y_predict)[0][-1] == 11):
                result = [np.argsort(y_predict)[0][-1]]
            else:
                result = [np.argsort(y_predict)[0][-num]]
                # print('選上的東西機率')
                # print(y_predict[0][result[0]])

            # result = [np.argsort(y_predict)[0][-2]]
            # print(result)
            # sys.exit(0)

        # print('看看')
        # print(result)

        # sys.exit(0)

        sentence += df.iloc[int(result[0])]['word']
    
    # 簡轉中
    sentence = cc.convert(sentence)
    generate_sentence.append(sentence)

print(generate_sentence)


