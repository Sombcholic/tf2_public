from bert.tokenization import bert_tokenization
import tensorflow as tf
import tensorflow_hub as hub
from keras.utils import to_categorical
from keras.models import Sequential
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from keras.models import load_model
import tensorflow_text 



# Build The Model
def build_model(bert_layer, preprocessor, labels_number, max_len=128):
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessed_text = preprocessor(text_input)
    outputs = bert_layer(preprocessed_text)

    # outputs['pooled_output'], outputs['sequence_output']
    
    # clf_output = outputs['sequence_output'][:, 0, :]
    clf_output = outputs['pooled_output']

    lay = tf.keras.layers.Dense(64, activation='relu', name='multiclass_layer1')(clf_output)
    lay2 = tf.keras.layers.Dense(64, activation='relu', name='multilabel_layer1')(clf_output)
    lay = tf.keras.layers.Dropout(0.2)(lay)
    lay = tf.keras.layers.Dense(32, activation='relu', name='multiclass_layer2')(lay)
    lay = tf.keras.layers.Dropout(0.2)(lay)

    lay2 = tf.keras.layers.Dropout(0.2)(lay2)
    out2 = tf.keras.layers.Dense(2000, activation='sigmoid', name='multilabel_output1')(lay2)
    
    concatenate1 = tf.keras.layers.Concatenate(axis=1, name='concatenate_layer1')([lay, out2])

    out = tf.keras.layers.Dense(labels_number, activation='softmax', name='main_output1')(concatenate1)

    model = tf.keras.models.Model(inputs=[text_input], outputs=[out, out2])
    model.compile(tf.keras.optimizers.Adam(lr=2e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

if __name__ == '__main__':
    bert_url = '../classification3/model/bert_zh_L-12_H-768_A-12_4'
    bert_layer = hub.KerasLayer(bert_url, trainable=True)

    preprocess_url  = '../classification3/model/bert_zh_preprocess_3'
    preprocessor = hub.KerasLayer(preprocess_url)

    model = build_model(bert_layer, preprocessor, 16, max_len=128)
    model.summary()
