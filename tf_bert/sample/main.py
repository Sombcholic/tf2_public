# https://ithelp.ithome.com.tw/articles/10306312?sc=rss.qu
import os
import sys
import shutil 
# 這兩個套件是為了要將訓練資料的路徑讀取工具也引進來

# os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1/bin")

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text # 前處理 
from official.nlp import optimization # 優化工具

import matplotlib.pyplot as plt # 畫圖用的套件

def test_tensorflow_gpu():
    print(tf.__version__)
    print(tf.test.gpu_device_name())
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# 引入訓練的資料
def import_data():
    if (os.path.exists('./aclImdb_v1.tar.gz') is False):
        url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
        dataset = tf.keras.utils.get_file('aclImdb_v1.tar.gz', url, untar=True, cache_dir='.', cache_subdir='')

        # 整合路徑
        dataset_dir = os.path.join(os.path.dirname(dataset), 'aclmdb')
        train_dir = os.path.join(dataset_dir, 'train')

        # 刪除沒有使用的資料夾
        remove_dir = os.path.join(train_dir, 'unsup')
        shutil.rmtree(remove_dir)

# 將資料轉換成訓練集 和 測試集
def trans_dataset():
    AUTOTUNE = tf.data.AUTOTUNE
    batch_size = 32
    seed = 42

    raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=seed)

    class_names = raw_train_ds.class_names
    train_ds = raw_train_ds.cache().prefetch(buffer_size=AUTOTUNE)

    val_ds = tf.keras.utils.text_dataset_from_directory(
        'aclImdb/train',
        batch_size=batch_size,
        validation_split=0.2,
        subset='validation',
        seed=seed)

    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    test_ds = tf.keras.utils.text_dataset_from_directory(
        'aclImdb/test',
        batch_size=batch_size)

    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
 
    return train_ds, val_ds, test_ds

# preprocess text
def preprocess_text(tfhub_handle_preprocess):
    bert_preprocess_model = hub.KerasLayer(tfhub_handle_preprocess) # 呼叫前處理模型
    text_test = ['this is such an amazing movie!'] # 先輸入簡單的句子看看會變成什麼樣子
    text_preprocessed = bert_preprocess_model(text_test)
    print(f'Keys       : {list(text_preprocessed.keys())}')
    print(f'Shape      : {text_preprocessed["input_word_ids"].shape}')
    print(f'Word Ids   : {text_preprocessed["input_word_ids"][0, :12]}')
    print(f'Input Mask : {text_preprocessed["input_mask"][0, :12]}')
    print(f'Type Ids   : {text_preprocessed["input_type_ids"][0, :12]}')

    return text_preprocessed

# 搭建模型
def build_classifier_model():
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
    return tf.keras.Model(text_input, net)

if __name__ == '__main__':
    test_tensorflow_gpu()
    
    print('\n\n\n')
    print('Lets Get Start!!!')
    
    # 引入訓練的資料
    import_data()

    # 將資料轉換成訓練集 和 測試資料集
    train_ds, val_ds, test_ds = trans_dataset()

    # 使用別人訓練好的model
    tfhub_handle_encoder = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1" # 這是預訓練模型
    tfhub_handle_preprocess = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3" # 這是 BERT 前處理需要用的模型

    text_preprocessed = preprocess_text(tfhub_handle_preprocess)

    sys.exit()

    # 將Input進行Encoding
    bert_model = hub.KerasLayer(tfhub_handle_encoder)
    bert_results = bert_model(text_preprocessed)
    print(f'Loaded BERT: {tfhub_handle_encoder}')
    print(f'Pooled Outputs Shape:{bert_results["pooled_output"].shape}')
    print(f'Pooled Outputs Values:{bert_results["pooled_output"][0, :12]}')
    print(f'Sequence Outputs Shape:{bert_results["sequence_output"].shape}')
    print(f'Sequence Outputs Values:{bert_results["sequence_output"][0, :12]}')


    # 建構模型
    classifier_model = build_classifier_model()

    # 設定loss,epochs, learning-rate
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    metrics = tf.metrics.BinaryAccuracy()

    epochs = 5
    steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1*num_train_steps)

    init_lr = 3e-5
    optimizer = optimization.create_optimizer(init_lr=init_lr,
                num_train_steps=num_train_steps,
                num_warmup_steps=num_warmup_steps,
                optimizer_type='adamw')

    # 開始訓練
    classifier_model.compile(optimizer=optimizer,
          loss=loss,
          metrics=metrics)
    print(f'Training model with {tfhub_handle_encoder}')
    history = classifier_model.fit(x=train_ds,
            validation_data=val_ds,
            epochs=epochs,
            verbose=1)     

    # 計算模型訓練效果
    loss, accuracy = classifier_model.evaluate(test_ds)
    print(f'Loss: {loss}')
    print(f'Accuracy: {accuracy}')     


    # 圖像化結果
    history_dict = history.history
    print(history_dict.keys())

    acc = history_dict['binary_accuracy']
    val_acc = history_dict['val_binary_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)
    fig = plt.figure(figsize=(10, 6))
    fig.tight_layout()

    plt.subplot(2, 1, 1)
    # r is for "solid red line"
    plt.plot(epochs, loss, 'r', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    # plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(epochs, acc, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
