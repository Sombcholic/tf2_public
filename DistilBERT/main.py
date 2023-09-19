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

def generate_data_to_model(train_data, train_labels):
    while True:
        yield (train_data, train_labels)

if __name__ == "__main__":
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
    model_name = '20230720-V2-generate_bert_model_epoch' + str(epochs)

    # Run the model
    # checkpoint = tf.keras.callbacks.ModelCheckpoint('./save_model/' + model_name + '.h5', monitor='val_multilabel_output1_accuracy', save_best_only=True, verbose=1)
    # checkpoint = tf.keras.callbacks.ModelCheckpoint('./save_model/' + model_name + '.h5', monitor='val_main_output1_accuracy', save_best_only=True, verbose=1)
    # earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1)
    # earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_multilabel_output1_loss', patience=3, verbose=1)
    # earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)

    train_sh = model.fit(
        train_data.content.values, train_labels,
        # validation_split=0.2,
        epochs=epochs,
        batch_size=16,
        verbose=1
    )

    model.save('./save_model/' + model_name + '.h5')
    

