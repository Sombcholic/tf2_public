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

def bert_encode(texts, tokenizer, max_len=512):
    all_tokens = []
    all_masks = []
    all_segments = []
    
    for text in texts:
        text = tokenizer.tokenize(text)
        
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len-len(input_sequence)
        
        tokens = tokenizer.convert_tokens_to_ids(input_sequence) + [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len
        
        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
        
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)

# Build a BERT layer
m_url = 'https://tfhub.dev/tensorflow/bert_zh_L-12_H-768_A-12/4'
bert_layer = hub.KerasLayer(m_url, trainable=True)

# Encoding the text
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = bert_tokenization.FullTokenizer(vocab_file, do_lower_case)

max_len = 250
# test_data = pd.read_csv('./data/Corona_NLP_test.csv', encoding='latin-1')

input_data = [
    ['我想查看看我的發票有沒有中獎', 0],
    ['想知道有什麼漫遊方案', 1],
    ['想知道有什麼國際漫遊方案', 1],
    ['我想知道中國漫遊方案', 1],
    ['起問有中國漫遊方案嗎', 1],
    ['請問我可以知道合約到期日嗎?', 2],
    ['合約資訊及優惠', 2],
    ['請給我帳單折抵說明', 3],
    ['請給我帳單折抵卷說明', 3 ],
    ['我想問一下如何使用帳單折抵卷？', 3],
    ['我希望了解帳單折抵卷的使用期限', 3],
    ['可以告訴我如何獲得帳單折抵卷嗎', 3],
    ['我想了解帳單折扣卷的使用規則。', 3],
    ['我想要查詢帳單折抵說明', 3],
    ['想知道合約到期日', 2],
    ['想知道合約到期日', 2],
    ['想查本期帳單', 4],
]


user_input = []
input_labels = []

for data in input_data:
    user_input.append(data[0].lower())
    input_labels.append(str(data[1]))

# test_input = bert_encode(user_input, tokenizer, max_len=max_len)


model_final = load_model('./save_model/Bert_classification3_20230618_V1_epoch50_bs4_myself.h5', custom_objects={'KerasLayer':hub.KerasLayer})

# # evaluate model 
# y_predict = model_final.predict(test_input)

y_predict = model_final.predict(user_input)


# # check results
# print(classification_report(test_input, y_predict)) 
print(y_predict)

result = np.argmax(y_predict, axis=1)

result_labels = []
wrong_input = []

i = 0
for r in result:
    probability = y_predict[i][r]

    if (probability >= 0.7):
        result_labels.append(str(r))
    else:
        result_labels.append('X')

    if (result_labels[i] != input_labels[i]):
        wrong_input.append([user_input[i], input_labels[i]])
        
    i += 1


print(result_labels)
print(input_labels)
print(wrong_input)


