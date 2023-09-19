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

    def get_df(self):
        self.df['content'] = self.df['content'].str.lower()
        return df

# Build The Model
def build_model(bert_layer, preprocessor, labels_number, max_len=128):
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessed_text = preprocessor(text_input)
    outputs = bert_layer(preprocessed_text)

    # outputs['pooled_output'], outputs['sequence_output']
    
    # clf_output = outputs['sequence_output'][:, 0, :]
    clf_output = outputs['pooled_output']
    
    lay = tf.keras.layers.Dense(64, activation='relu')(clf_output)
    lay = tf.keras.layers.Dropout(0.2)(lay)
    out = tf.keras.layers.Dense(labels_number, activation='sigmoid')(lay)
    # out = tf.keras.layers.Dense(labels_number, activation='softmax')(lay)
    
    model = tf.keras.models.Model(inputs=[text_input], outputs=out)
    # model.compile(tf.keras.optimizers.Adam(lr=2e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(tf.keras.optimizers.Adam(lr=2e-5), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


if __name__ == "__main__":

    data_name = '20230724-V1-云南省昆明市西山区昆州路'

    df = pd.read_csv('./data/' + data_name + '.csv', encoding='utf-8')
    dataManager = DataManager(data_name, df)


    df = dataManager.get_df()

    train_data, test_data = train_test_split(df, test_size=0.2)

    # Label encoding of labels
    label = preprocessing.LabelEncoder()
    y = label.fit_transform(train_data['label'])
    y = to_categorical(y)


    print(type(y[0][0]))
    sys.exit(0)

    # 將input轉換成小寫
    train_data['content'] = train_data['content'].str.lower()

    # Chinese version
    bert_url = './model/distilbert_multi_cased_L-6_H-768_A-12_1'
    bert_layer = hub.KerasLayer(bert_url, trainable=True)

    preprocess_url  = './model/distilbert_multi_cased_preprocess_2'
    preprocessor = hub.KerasLayer(preprocess_url)

    train_labels = y
    labels = label.classes_

    print(train_labels[0])
    sys.exit(0)

    max_len = 128

    model = build_model(bert_layer, preprocessor, len(labels), max_len=max_len)
    model.summary()

    epochs = 20
    model_name = '20230725-V1-hospital_bert_model_epoch' + str(epochs)

    # Run the model
    checkpoint = tf.keras.callbacks.ModelCheckpoint('./save_model/' + model_name + '.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
    earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    # earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)

    train_sh = model.fit(
        train_data.content.values, train_labels,
        validation_split=0.2,
        epochs=epochs,
        callbacks=[checkpoint, earlystopping],
        batch_size=16,
        verbose=1
    )


    model.save('./save_model/' + model_name + '.h5')

