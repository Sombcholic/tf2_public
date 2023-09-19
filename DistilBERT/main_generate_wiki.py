# https://www.kaggle.com/code/nayansakhiya/text-classification-using-bert/notebook
# https://zhuanlan.zhihu.com/p/35866604 -> 嘗試解決OOM
# import tokenization
# 解析wiki https://sfhsu29.medium.com/nlp-%E5%B0%88%E6%AC%84-1-2-%E5%A6%82%E4%BD%95%E8%A8%93%E7%B7%B4%E8%87%AA%E5%B7%B1%E7%9A%84-word2vec-5a0754c5cb09
# 高質量文本 https://github.com/CLUEbenchmark/CLUECorpus2020/
# 高質量文本 https://github.com/JunnYu/FLASHQuad_pytorch

from bert.tokenization import bert_tokenization
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.losses import BinaryFocalCrossentropy
from tensorflow.keras.layers import LeakyReLU
from keras.utils import to_categorical
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from keras.models import load_model
import tensorflow_text as text
import time
import yaml
# from lxml import etree
from gensim.corpora import WikiCorpus


class DataManager():
    def __init__(self, data_name, df):
        self.data_name = data_name
        self.df = df

    def create_sentence(self):
        # print(self.df['content'])
        # train_data['label']
        "[CLS]"

        label_content = []

        self.df['content'] = self.df['content'].str.lower()

        for sentence in self.df['content']:
            sentence = sentence.replace(' ', '')

            for i in range(len(sentence)):
                try:
                    label_content.append([sentence[:i+1], sentence[i+1]])
                except:
                    label_content.append([sentence[:i+1], '[CLS]'])

        df = pd.DataFrame(label_content, columns=['content', 'label'])
        
        return df


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

# Build The Model
def build_model(bert_layer, preprocessor, labels_number, max_len=128, max_keyword_len = 0):
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessed_text = preprocessor(text_input)
    outputs = bert_layer(preprocessed_text)

    # outputs['pooled_output'], outputs['sequence_output']
    # clf_output = outputs['sequence_output'][:, 0, :]
    clf_output = outputs['pooled_output']
    
    lay = tf.keras.layers.Dense(32, activation='relu')(clf_output)
    lay = tf.keras.layers.Dropout(0.2)(lay)
    # lay = tf.keras.layers.Dense(32, activation='relu')(lay)
    # lay = tf.keras.layers.Dropout(0.2)(lay)
    out = tf.keras.layers.Dense(labels_number, activation='softmax')(lay)
    
    model = tf.keras.models.Model(inputs=[text_input], outputs=out)
    model.compile(tf.keras.optimizers.Adam(lr=2e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model


if __name__ == "__main__":
    input_filename = './data/zhwiki-20230501-pages-articles-multistream.xml.bz2'
    output_filename = 'wiki-preprocessed-lemma.txt'

    wiki = WikiCorpus(input_filename, dictionary={})
    for text in wiki.get_texts():
        # str_line = bytes.join(b' ', text).decode()
        print(text)