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


class DataManager():
    def __init__(self, data_name, df):
        self.data_name = data_name
        self.df = df

    def get_df(self):
        self.df['content'] = self.df['content'].str.lower()
        return df

# Build The Model
def build_model(bert_layer, preprocessor, label_len, learning_rate, max_len=128, sick_len = 0):
    # Jaccard Loss 可以用
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessed_text = preprocessor(text_input)
    outputs = bert_layer(preprocessed_text)

    clf_output = outputs['pooled_output']

    c_lay = tf.keras.layers.Dense(64, activation='relu', name='multiclass_layer1')(clf_output)
    l_lay = tf.keras.layers.Dense(64, activation='relu', name='multilabel_layer1')(clf_output)
    c_lay = tf.keras.layers.Dropout(0.2)(c_lay)
    l_lay = tf.keras.layers.Dropout(0.2)(l_lay)
    
    c_lay = tf.keras.layers.Dense(32, activation='relu', name='multiclass_layer2')(c_lay)
    c_lay = tf.keras.layers.Dropout(0.2)(c_lay)
    # l_lay = tf.keras.layers.Dense(32, activation='relu', name='multilabel_layer2')(l_lay)
    l_lay = tf.keras.layers.Dense(32, activation=LeakyReLU(alpha=0.3), name='multilabel_layer2')(l_lay)
    l_lay = tf.keras.layers.Dropout(0.2)(l_lay)

    out2 = tf.keras.layers.Dense(sick_len, activation='sigmoid', name='multilabel_output1')(l_lay)
    # l_lay = tf.keras.layers.Dense(32, activation='relu', name='out2_label')(out2)

    concatenate1 = tf.keras.layers.Concatenate(name='concatenate_layer1')([c_lay, l_lay])

    out = tf.keras.layers.Dense(label_len, activation='sigmoid', name='main_output1')(concatenate1)
    
    # https://www.tensorflow.org/api_docs/python/tf/keras/losses/BinaryFocalCrossentropy
    focal_loss = BinaryFocalCrossentropy(gamma=4.0, alpha=0.75)

    model = tf.keras.models.Model(inputs=[text_input], outputs=[out, out2])
    # model.compile(tf.keras.optimizers.Adam(lr=2e-5), loss=['categorical_crossentropy', 'binary_crossentropy'], metrics=['accuracy'], loss_weights = [0.2, 0.8])
    # model.compile(tf.keras.optimizers.Adam(lr=2e-5), loss=[focal_loss, focal_loss], metrics=['accuracy'], loss_weights = [0.1, 0.9])
    model.compile(tf.keras.optimizers.Adam(lr=learning_rate), loss=[focal_loss, focal_loss], metrics=['accuracy'], loss_weights = [0.1, 0.9])
    # model.compile(tf.keras.optimizers.Adam(lr=2e-5), loss=[jaccard_loss, jaccard_loss], metrics=['accuracy'], loss_weights = [0.1, 0.9])
    # model.compile(tf.keras.optimizers.Adam(lr=learning_rate), loss=[jaccard_loss, jaccard_loss], metrics=['accuracy'], loss_weights = [0.1, 0.9])
    
    return model

def preprocess_label(df):
    labelList = []
    sickList = []

    new_dataset = []


    for index, row in df.iterrows():
        label = row['label']
        label = label.replace("'", '').replace('[', '').replace(']', '')

        _labelList = label.split(',')

        for l in _labelList:
            if (l not in labelList and l != ''):
                labelList.append(l)

        if (row['sick'] not in sickList):
            sickList.append(row['sick'])

    for index, row in df.iterrows():
        data = []

        # 找看看有沒有label
        _label = np.zeros(len(labelList), dtype=np.float32)

        for label in labelList:
            if (row['label'].find(label) >= 0):
                _label[labelList.index(label)] = 1.0

        _sick = np.zeros(len(sickList), dtype=np.float32)
        _sick[sickList.index(row['sick'])] = 1.0

        new_dataset.append([_label, _sick, row['content']])

    new_df = pd.DataFrame(new_dataset, columns=['label', 'sick', 'content'])

    return new_df, labelList, sickList

# def jaccard_loss_exclusive(y_true, y_pred):
#     # 計算相斥標籤之間的 Jaccard Loss
#     intersection = tf.reduce_sum(y_true * y_pred, axis=-1)
#     union = tf.reduce_sum(tf.clip_by_value(y_true + y_pred, 0, 1), axis=-1)
#     jaccard = (intersection + 1e-5) / (union + 1e-5)
#     # 只保留相斥標籤的損失
#     exclusive_jaccard_loss = 1 - jaccard * (1 - y_true)
#     # 平均損失
#     num_samples = tf.cast(tf.shape(y_true)[0], dtype=tf.float32)
#     exclusive_jaccard_loss = tf.reduce_sum(exclusive_jaccard_loss) / num_samples
#     return exclusive_jaccard_loss

def jaccard_loss(y_true, y_pred):
    # 計算交集和聯集的元素數量
    intersection = tf.reduce_sum(y_true * y_pred, axis=-1)
    union = tf.reduce_sum(tf.clip_by_value(y_true + y_pred, 0, 1), axis=-1)
    # 計算 Jaccard 相似性係數
    jaccard = (intersection + 1e-5) / (union + 1e-5)
    # 計算 Jaccard Loss
    jaccard_loss = 1 - jaccard
    # 平均損失
    num_samples = tf.cast(tf.shape(y_true)[0], dtype=tf.float32)
    jaccard_loss = tf.reduce_mean(jaccard_loss) / num_samples
    return jaccard_loss


if __name__ == "__main__":

    data_name = '20230725_V1_hospital'

    df = pd.read_csv('./data/' + data_name + '.csv', encoding='utf-8')
    dataManager = DataManager(data_name, df)


    df = dataManager.get_df()

    df, labelList, sickList = preprocess_label(df)

    # print(df['label'])
    # sys.exit(0)

    try:
        with open('./data/' + data_name + 'labelList.yml', 'w', encoding='utf-8') as f:
            yaml.dump(labelList, f, allow_unicode=True)
    except Exception as e:
        print(e)
        ...

    try:
        with open('./data/' + data_name + 'sickList.yml', 'w', encoding='utf-8') as f:
            yaml.dump(sickList, f, allow_unicode=True)
    except Exception as e:
        print(e)
        ...

    # train_data, test_data = train_test_split(df, test_size=0.2)
    
    # 將input轉換成小寫
    # train_data['content'] = train_data['content'].str.lower()

    # 切割驗證
    train_data, val_data = train_test_split(df, test_size=0.2)

    train_data['content'] = train_data['content'].str.lower()
    val_data['content'] = val_data['content'].str.lower()

    # Chinese version
    bert_url = './model/distilbert_multi_cased_L-6_H-768_A-12_1'
    bert_layer = hub.KerasLayer(bert_url, trainable=True)

    preprocess_url  = './model/distilbert_multi_cased_preprocess_2'
    preprocessor = hub.KerasLayer(preprocess_url)

    max_len = 128
    # learning_rate = 0.0002
    learning_rate = 0.0005
    epochs = 20
    early_stop = 5
    model_name = '20230804-V1-hospital_bert_model_epoch' + str(epochs) + "_early" + str(early_stop) + "_jaccard_loss_" + str(learning_rate)
    model = build_model(bert_layer, preprocessor, len(labelList), learning_rate, max_len=max_len, sick_len = len(sickList))
    model.summary()

    # Run the model
    checkpoint = tf.keras.callbacks.ModelCheckpoint('./save_model/' + model_name + '.h5', monitor='val_multilabel_output1_accuracy', save_best_only=True, verbose=1)
    earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_multilabel_output1_loss', patience=early_stop, verbose=1)
    # earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_multilabel_output1_accuracy', patience=early_stop, verbose=1)
    # earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_multilabel_output1_loss', patience=3, verbose=1)

    # labels = K.cast_to_floatx(train_data.label.values)
    # sicks = K.cast_to_floatx(train_data.sick.values)

    # labels = tf_data = tf.constant(train_data.label.values)
    # sicks = tf_data = tf.constant(train_data.sick.values)

    train_labels = []
    train_sicks = []
    val_labels = []
    val_sicks = []

    for i in train_data.label.values:
        train_labels.append(i)

    for i in train_data.sick.values:
        train_sicks.append(i)

    for i in val_data.label.values:
        val_labels.append(i)

    for i in val_data.sick.values:
        val_sicks.append(i)


    train_labels = np.array(train_labels)
    train_sicks = np.array(train_sicks)

    val_labels = np.array(val_labels)
    val_sicks = np.array(val_sicks)

    y_combine = (train_labels, train_sicks)
    val_y_combine = (val_labels, val_sicks)

    with tf.device("CPU"):
        dataset = tf.data.Dataset.from_tensor_slices((train_data.content.values, y_combine))
        dataset = dataset.shuffle(buffer_size=1024).batch(16)

        val_dataset = tf.data.Dataset.from_tensor_slices((val_data.content.values, val_y_combine))
        val_dataset = val_dataset.batch(16)

    train_sh = model.fit(
        dataset, 
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[checkpoint, earlystopping],
        verbose=2
    )

    # train_sh = model.fit(
    #     train_data.content.values, [train_labels, train_sicks],
    #     # train_data.content.values, [train_data.label.values, train_data.sick.values],
    #     validation_split=0.2,
    #     epochs=epochs,
    #     callbacks=[checkpoint, earlystopping],
    #     batch_size=8,
    #     verbose=2
    # )


    model.save('./save_model/' + model_name + '.h5')

