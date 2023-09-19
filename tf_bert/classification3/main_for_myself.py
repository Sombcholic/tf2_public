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

df = pd.read_csv('./data/20230617-Test1-ALL.csv', encoding='utf-8')
# test_data = pd.read_csv('./data/data1.csv', encoding='utf-8')

train_data, test_data = train_test_split(df, test_size=0.2)

# Label encoding of labels
label = preprocessing.LabelEncoder()
y = label.fit_transform(train_data['label'])
y = to_categorical(y)
print(y[:5])

# Build a BERT layer
# English version
m_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2'
bert_layer = hub.KerasLayer(m_url, trainable=True)
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = bert_tokenization.FullTokenizer(vocab_file, do_lower_case)


# Chinese version
# m_url = 'https://tfhub.dev/tensorflow/bert_zh_L-12_H-768_A-12/4'
# bert_layer = hub.KerasLayer(m_url, trainable=True)

# # Encoding the text
# vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
# do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
# tokenizer = bert_tokenization.FullTokenizer(vocab_file, do_lower_case)

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
def build_model(bert_layer, max_len=512):
    input_word_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")
    
    pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    
    clf_output = sequence_output[:, 0, :]
    
    lay = tf.keras.layers.Dense(64, activation='relu')(clf_output)
    lay = tf.keras.layers.Dropout(0.2)(lay)
    lay = tf.keras.layers.Dense(32, activation='relu')(lay)
    lay = tf.keras.layers.Dropout(0.2)(lay)
    out = tf.keras.layers.Dense(5, activation='softmax')(lay)
    
    model = tf.keras.models.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    model.compile(tf.keras.optimizers.Adam(lr=2e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model


max_len = 250
train_input = bert_encode(train_data.content.values, tokenizer, max_len=max_len)
test_input = bert_encode(test_data.content.values, tokenizer, max_len=max_len)

train_labels = y

labels = label.classes_

model = build_model(bert_layer, max_len=max_len)
model.summary()

# Run the model
checkpoint = tf.keras.callbacks.ModelCheckpoint('Bert_classification3_20230617_V1_epoch50_bs4_myself.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)

train_sh = model.fit(
    train_input, train_labels,
    validation_split=0.2,
    epochs=50,
    callbacks=[checkpoint, earlystopping],
    batch_size=4,
    verbose=1
)

model.save('Bert_classification3_20230617_V1_epoch50_bs4_myself.h5')

