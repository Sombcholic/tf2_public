# https://www.kaggle.com/code/nayansakhiya/text-classification-using-bert/notebook
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
import copy


class DataManager():
    def __init__(self, data_name):
        self.data_name = data_name
        self.keywordDict = {}
        self.keyword_len = 0
        self.sentence = []
        self.keyword_poisition = {}
        self.keyword_index = {}
        self.keyword_sentence1 = None 
        self.keyword_sentence2 = None 

        self._load_data()
        self.create_keyword_index()
        self.create_keyword_position()
        
    def _load_data(self):
        try:
            with open('./data/keyword/keywordDict.yml', 'r', encoding="utf-8") as f:
                self.keywordDict = yaml.safe_load(f)
                self.keyword_len = len(self.keywordDict)
        except Exception as e:
            print(e)
            ...

        try:
            with open('./data/keyword/20230626-Test1-class16-sheet0.yml', 'r', encoding="utf-8") as f:
                self.keyword_sentence1 = yaml.safe_load(f)
        except Exception as e:
            print(e)
            ...

        try:
            with open('./data/keyword/20230626-Test1-class16-sheet1.yml', 'r', encoding="utf-8") as f:
                self.keyword_sentence2 = yaml.safe_load(f)
        except Exception as e:
            print(e)
            ...

    def create_keyword_index(self):
        i = 0
        for key, value in self.keywordDict.items():
            self.keyword_index[key] = i
            
            for v in value:
                self.keyword_index[v] = i

            i += 1

        try:
            with open('./data/keyword/' + self.data_name + '.yml', 'w', encoding='utf-8') as f:
                yaml.dump(self.keyword_index, f, allow_unicode=True)
        except Exception as e:
            print(e)
            ...

    def create_keyword_position(self):

        keyword_sentence_list = [self.keyword_sentence1, self.keyword_sentence2]

        for keyword_sentence in keyword_sentence_list:
            for key, value in keyword_sentence.items():
                
                position = []
                for keyword in value['keyword_list']:
                    try:
                        if (self.keyword_index.get(keyword) is not None):
                            position.append(self.keyword_index.get(keyword))
                    except:
                        ...

                for sentence in value['sentence']:
                    try:
                        if (self.keyword_poisition.get(sentence) is None):
                            self.keyword_poisition[sentence] = position
                    except:
                        ...

    def encode_keyword(self, sentenceList):
        output = []

        for sentence in sentenceList:
            key_position = np.zeros(self.keyword_len)
            # key_position = [0 for i in range(self.keyword_len)]

            if (self.keyword_poisition.get(sentence) is not None):
                position = self.keyword_poisition.get(sentence)
                for p in position:
                    key_position[p] = 1

                # output.append([sentence, key_position])
                output.append(key_position)
            else:
                print('這句話沒有找到相符的keyword')
                print(sentence)

        # output = pd.DataFrame(output, columns=['sentence', 'key_position'])

        return np.array(output)


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

    c_lay = tf.keras.layers.Dense(64, activation='relu', name='multiclass_layer1')(clf_output)
    l_lay = tf.keras.layers.Dense(64, activation='relu', name='multilabel_layer1')(clf_output)
    c_lay = tf.keras.layers.Dropout(0.2)(c_lay)
    l_lay = tf.keras.layers.Dropout(0.2)(l_lay)
    
    c_lay = tf.keras.layers.Dense(32, activation='relu', name='multiclass_layer2')(c_lay)
    c_lay = tf.keras.layers.Dropout(0.2)(c_lay)    
    # l_lay = tf.keras.layers.Dense(32, activation='relu', name='multilabel_layer2')(l_lay)
    l_lay = tf.keras.layers.Dense(32, activation=LeakyReLU(alpha=0.3), name='multilabel_layer2')(l_lay)
    l_lay = tf.keras.layers.Dropout(0.2)(l_lay)

    out2 = tf.keras.layers.Dense(max_keyword_len, activation='sigmoid', name='multilabel_output1')(l_lay)
    # l_lay = tf.keras.layers.Dense(32, activation='relu', name='out2_label')(out2)

    concatenate1 = tf.keras.layers.Concatenate(name='concatenate_layer1')([c_lay, l_lay])

    out = tf.keras.layers.Dense(labels_number, activation='softmax', name='main_output1')(concatenate1)
    

    # https://www.tensorflow.org/api_docs/python/tf/keras/losses/BinaryFocalCrossentropy
    # 若傾向預測出1，嘗試將alpha變小
    # 若傾向預測出0，嘗試將alpha變大
    # L(p, y) = - α * (1 - p) ^ γ * y * log(p) - (1 - α) * p ^ γ * (1 - y) * log(1 - p)
    # L(p, y) 是二元焦點交叉熵損失函數。
    # p 是模型預測的機率值，表示該樣本屬於正類的機率。
    # y 是真實的標籤值，如果樣本屬於正類，則為1；如果樣本屬於負類，則為0。
    # α 是焦點參數（focusing parameter），控制正負樣本的權重。當α > 0.5時，正樣本（y=1）的損失權重大於負樣本（y=0）；當α < 0.5時，正樣本的損失權重小於負樣本；當α = 0.5時，兩者的權重相等。
    # γ 是焦點指數（focusing exponent），控制焦點損失的形狀。當γ > 0時，易分樣本的損失相對減少，難分樣本的損失相對增加；當γ = 0時，焦點損失相當於標準的交叉熵損失。
    # alpha 預設是0.25
    focal_loss = BinaryFocalCrossentropy(gamma=4.0, alpha=0.75)

    model = tf.keras.models.Model(inputs=[text_input], outputs=[out, out2])
    # model.compile(tf.keras.optimizers.Adam(lr=2e-5), loss=['categorical_crossentropy', 'binary_crossentropy'], metrics=['accuracy'], loss_weights = [0.2, 0.8])
    model.compile(tf.keras.optimizers.Adam(lr=2e-5), loss=['categorical_crossentropy', focal_loss], metrics=['accuracy'], loss_weights = [0.1, 0.9])
    
    return model


if __name__ == "__main__":

    data_name = '20230620-Test1-class16'

    df = pd.read_csv('./data/' + data_name + '.csv', encoding='utf-8')

    # print(len(df))

    df2 = df.copy()
    df3 = df.copy()
    df4 = df.copy()
    df5 = df.copy()
    df6 = df.copy()
    df7 = df.copy()

    df = pd.concat([df, df2, df3, df4, df5, df6, df7])

    # print(len(df))

    # sys.exit(0)

    train_data, test_data = train_test_split(df, test_size=0.2)

    dataManager = DataManager(data_name)
    train_keyword_data = dataManager.encode_keyword(train_data.content.values)
    test_keyword_data = dataManager.encode_keyword(test_data.content.values)

    # Label encoding of labels
    label = preprocessing.LabelEncoder()
    y = label.fit_transform(train_data['label'])
    y = to_categorical(y)

    # 將input轉換成小寫
    train_data['content'] = train_data['content'].str.lower()

    # Chinese version
    bert_url = './model/distilbert_multi_cased_L-6_H-768_A-12_1'
    # bert_url = './model/bert_zh_L-12_H-768_A-12_4'
    bert_layer = hub.KerasLayer(bert_url, trainable=True)

    preprocess_url  = './model/distilbert_multi_cased_preprocess_2'
    # preprocess_url  = './model/bert_zh_preprocess_3'
    preprocessor = hub.KerasLayer(preprocess_url)
    

    max_len = 128
    train_labels = y

    labels = label.classes_

    model = build_model(bert_layer, preprocessor, len(labels), max_len=max_len, max_keyword_len = dataManager.keyword_len)
    model.summary()

    epochs = 10
    model_name = '20230717-V1-class16-mix_model-binaryFocal_gamma4_alpha025_connectlabel_retrain_epoch' + str(epochs) + '_bs4_myself'

    # Run the model
    checkpoint = tf.keras.callbacks.ModelCheckpoint('./save_model/' + model_name + '.h5', monitor='val_multilabel_output1_accuracy', save_best_only=True, verbose=1)
    # checkpoint = tf.keras.callbacks.ModelCheckpoint('./save_model/' + model_name + '.h5', monitor='val_main_output1_accuracy', save_best_only=True, verbose=1)
    # earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1)
    earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_multilabel_output1_loss', patience=3, verbose=1)
    # earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)

    start = time.time()
    for e in range(epochs):
        split = 4
        for i in range(split):
            print('目前第一次stepRange -> ', i)
            if (i == 0):
                data = train_data.iloc[:round(len(train_data)/split)]
                labels = train_labels[:round(len(train_labels)/split)]
                keyword_data = train_keyword_data[:round(len(train_keyword_data)/split)]

            elif (i == split-1):
                # model = load_model('./save_model/' + model_name + '.h5', custom_objects={'KerasLayer':hub.KerasLayer})

                data = train_data.iloc[round(len(train_data)/split)*i:]
                labels = train_labels[round(len(train_labels)/split)*i:]
                keyword_data = train_keyword_data[round(len(train_keyword_data)/split)*i:]
            else:
                # model = load_model('./save_model/' + model_name + '.h5', custom_objects={'KerasLayer':hub.KerasLayer})

                data = train_data.iloc[round(len(train_data)/split)*i:round(len(train_data)/split)*(i+1)]
                labels = train_labels[round(len(train_data)/split)*i:round(len(train_labels)/split)*(i+1)]
                keyword_data = train_keyword_data[round(len(train_data)/split)*i:round(len(train_keyword_data)/split)*(i+1)]

            train_sh = model.fit(
                data.content.values, [labels, keyword_data],
                validation_split=0.2,
                epochs=1,
                batch_size=16,
                verbose=1
            )

            model.save('./save_model/' + model_name + '.h5')
    
    end = time.time()

    print(end - start)

