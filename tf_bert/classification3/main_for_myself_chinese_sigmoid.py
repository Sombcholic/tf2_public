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
import time

df = pd.read_csv('./data/20230620-Test1-class16.csv', encoding='utf-8')
# test_data = pd.read_csv('./data/data1.csv', encoding='utf-8')

train_data, test_data = train_test_split(df, test_size=0.2)

# Label encoding of labels
label = preprocessing.LabelEncoder()
y = label.fit_transform(train_data['label'])
y = to_categorical(y)

# 將input轉換成小寫
train_data['content'] = train_data['content'].str.lower()

# Build a BERT layer
# English version
# m_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2'
# bert_layer = hub.KerasLayer(m_url, trainable=True)
# vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
# do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
# tokenizer = bert_tokenization.FullTokenizer(vocab_file, do_lower_case)


# Chinese version
# m_url = 'https://tfhub.dev/tensorflow/bert_zh_L-12_H-768_A-12/4'
bert_url = './model/bert_zh_L-12_H-768_A-12_4'
bert_layer = hub.KerasLayer(bert_url, trainable=True)

# preprocess_url  = 'https://tfhub.dev/tensorflow/bert_zh_preprocess/3'
preprocess_url  = './model/bert_zh_preprocess_3'
# preprocessor = hub.KerasLayer(preprocess_url)
preprocessor = hub.KerasLayer(preprocess_url)

# aa = preprocessor(['我想吃飯', '他想要睡覺'])
# print(aa)

# sys.exit(0)

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
def build_model(bert_layer, preprocessor, labels_number, max_len=128):
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessed_text = preprocessor(text_input)
    outputs = bert_layer(preprocessed_text)

    # outputs['pooled_output'], outputs['sequence_output']
    
    # clf_output = outputs['sequence_output'][:, 0, :]
    clf_output = outputs['pooled_output']
    
    lay = tf.keras.layers.Dense(64, activation='relu')(clf_output)
    lay = tf.keras.layers.Dropout(0.2)(lay)
    lay = tf.keras.layers.Dense(32, activation='relu')(lay)
    lay = tf.keras.layers.Dropout(0.2)(lay)
    out = tf.keras.layers.Dense(labels_number, activation='sigmoid')(lay)
    # out = tf.keras.layers.Dense(labels_number, activation='softmax')(lay)
    
    model = tf.keras.models.Model(inputs=[text_input], outputs=out)
    # model.compile(tf.keras.optimizers.Adam(lr=2e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(tf.keras.optimizers.Adam(lr=2e-5), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

max_len = 128
train_labels = y


labels = label.classes_

model = build_model(bert_layer, preprocessor, len(labels), max_len=max_len)
model.summary()

model_name = '20230628-V2-class16-sigmoid_epoch20_bs4_myself'

# Run the model
checkpoint = tf.keras.callbacks.ModelCheckpoint('./save_model/' + model_name + '.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1)
# earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)

train_sh = model.fit(
    train_data.content.values, train_labels,
    validation_split=0.2,
    epochs=20,
    callbacks=[checkpoint, earlystopping],
    batch_size=4,
    verbose=1
)

model.save('./save_model/' + model_name + '.h5')

