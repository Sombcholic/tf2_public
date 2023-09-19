# https://www.analyticsvidhya.com/blog/2021/12/text-classification-using-bert-and-tensorflow/
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import ipykernel
import sys
import joblib 
import numpy as np


if __name__ == '__main__':
    df= pd.read_excel('spam1.xlsx')
    # print(df.head())

    # print(df['label'])c

    df['label']=df['label'].apply(lambda x: 1 if x=='spam' else 0)
    # print(df.head())

    X_train, X_test, y_train, y_test = train_test_split(df['message'],df['label'])

    bert_preprocess = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3')
    bert_encoder = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4')

    # Bert layers
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessed_text = bert_preprocess(text_input)
    outputs = bert_encoder(preprocessed_text)

    sys.exit(0)
    print('這裡唷唷')
    # Neural network layers
    l = tf.keras.layers.Dropout(0.1, name="dropout")(outputs['pooled_output'])
    l = tf.keras.layers.Dense(1, activation='sigmoid', name="output")(l)
    # Use inputs and outputs to construct a final model
    print('到這裡1')
    model = tf.keras.Model(inputs=[text_input], outputs = [l])

    print('到這裡2')
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    print('到這裡3')
    print(X_train[0])
    print(y_train[0])

    y_train = y_train.astype(np.int32)

    # X_train = tf.constant(np.array(X_train))
    # y_train = tf.constant(np.array(y_train))

    model.fit(X_train, y_train, epochs=2, batch_size = 32)

    print('到這裡4')

    # filename = 'Bert_sigmoid_class.sav'
    # joblib.dump(model, filename)

    # y_predicted = model.predict(X_test)
    # y_predicted = y_predicted.flatten()
    # print(y_predicted)


    # load model with joblib
    # loaded_model = joblib.load(filename)

    # # evaluate model 
    # y_predict = model.predict(X_test)

    # # check results
    # print(classification_report(y_test, y_predict)) 




