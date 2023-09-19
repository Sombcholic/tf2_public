# https://www.kaggle.com/code/nayansakhiya/text-classification-using-bert/notebook
# https://zhuanlan.zhihu.com/p/35866604 -> 嘗試解決OOM
# import tokenization
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
from keras import backend as K
from tensorflow.compat.v1.keras.backend import set_session
from tensorflow.compat.v1.keras.backend import get_session
from tensorflow.compat.v1.keras.backend import clear_session


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

# Reset Keras Session
def clear_tf(sess):
    sess.close()
    tf.compat.v1.reset_default_graph()

def set_tf(percent = 0):
    config = tf.compat.v1.ConfigProto()
    if (percent  != 0):
        config.gpu_options.per_process_gpu_memory_fraction = 0.8
    sess = tf.compat.v1.Session(config=config)
    set_session(sess)

    return sess


if __name__ == "__main__":

    # config = tf.compat.v1.ConfigProto()
    # config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.8
    # sess = tf.compat.v1.Session(config=config)
    # set_session(sess)

    sess = set_tf(0.8)

    data_name = '20230620-Test1-class16'

    df = pd.read_csv('./data/' + data_name + '.csv', encoding='utf-8')
    dataManager = DataManager(data_name, df)
    df = dataManager.create_sentence()

    train_data, test_data = train_test_split(df, test_size=0.2)

    # Label encoding of labels
    label = preprocessing.LabelEncoder()
    label_index = label.fit_transform(train_data['label'])
    label_word = label.classes_

    label_encoding = to_categorical(label_index)

    label_dict = []

    i = 0
    for word in label_word:
        label_dict.append([word, label_index[i]])
        i += 1

    try:
        with open('./data/keyword/' + data_name + '_generate_dict.yml', 'w', encoding='utf-8') as f:
            yaml.dump(label_dict, f, allow_unicode=True)
    except Exception as e:
        print(e)
        ...

    df2 = pd.DataFrame(label_dict, columns=['word', 'index'])
    df2.to_csv('./data/keyword/' + data_name + '_generate_dict.csv', encoding='utf8')  

    df2 = pd.read_csv('./data/keyword/' + data_name + '_generate_dict.csv', encoding='utf-8')

    # 將input轉換成小寫
    train_data['content'] = train_data['content'].str.lower()

    # Chinese version
    bert_url = './model/distilbert_multi_cased_L-6_H-768_A-12_1'
    bert_layer = hub.KerasLayer(bert_url, trainable=True)

    preprocess_url  = './model/distilbert_multi_cased_preprocess_2'
    preprocessor = hub.KerasLayer(preprocess_url)

    max_len = 128
    train_labels = label_encoding

    model = build_model(bert_layer, preprocessor, len(label_word), max_len=max_len, max_keyword_len = 128)
    model.summary()

    epochs = 5
    model_name = '20230717-V1-generate_distilbert_model_retrain_epoch' + str(epochs)

    del model
    for i in range(100000000000000):
        print(i)

    
    for e in range(epochs):
        split = 4
        start = time.time()
        for i in range(split):
            print('目前第一次stepRange -> ', i)
            if (i == 0 and e == 0):
                data = train_data.iloc[:round(len(train_data)/split)]
                labels = train_labels[:round(len(train_labels)/split)]
            
            elif (i == 0):
                sess = set_tf(0)
                model = load_model('./save_model/20230717-V1-generate_distilbert_model_retrain_epoch5.h5', custom_objects={'KerasLayer':hub.KerasLayer})
                data = train_data.iloc[:round(len(train_data)/split)]
                labels = train_labels[:round(len(train_labels)/split)]

            elif (i == split-1):
                # model = load_model('./save_model/' + model_name + '.h5', custom_objects={'KerasLayer':hub.KerasLayer})
                # reset tensorflow memory
                sess = set_tf(0)
                model = load_model('./save_model/20230717-V1-generate_distilbert_model_retrain_epoch5.h5', custom_objects={'KerasLayer':hub.KerasLayer})
                data = train_data.iloc[round(len(train_data)/split)*i:]
                labels = train_labels[round(len(train_labels)/split)*i:]
            else:
                # model = load_model('./save_model/' + model_name + '.h5', custom_objects={'KerasLayer':hub.KerasLayer})
                # reset tensorflow memory
                sess = set_tf(0)
                model = load_model('./save_model/20230717-V1-generate_distilbert_model_retrain_epoch5.h5', custom_objects={'KerasLayer':hub.KerasLayer})
                data = train_data.iloc[round(len(train_data)/split)*i:round(len(train_data)/split)*(i+1)]
                labels = train_labels[round(len(train_data)/split)*i:round(len(train_labels)/split)*(i+1)]

            train_sh = model.fit(
                data.content.values, labels,
                epochs=1,
                batch_size=16,
                verbose=1
            )

            model.save('./save_model/' + model_name + '.h5')
            del model

            # sess.close()
            # tf.compat.v1.reset_default_graph()
            # K.clear_session()
            # del model
            # reset_keras()
            clear_tf(sess)

            for i in range(100000000000000):
                print(i)
    
            end = time.time()
            print(end - start)

    

