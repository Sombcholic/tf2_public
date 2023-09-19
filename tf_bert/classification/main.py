# https://www.kaggle.com/code/nkaenzig/bert-tensorflow-2-huggingface-transformers/notebook
# from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from transformers import TFBertModel
from tensorflow.keras.layers import Dense, Flatten
import time
from transformers import create_optimizer

def tokenize_sentences(sentences, tokenizer, max_seq_len = 128):
    tokenized_sentences = []

    for sentence in sentences:
        tokenized_sentence = tokenizer.encode(
            sentence,                  # Sentence to encode.
            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
            max_length = max_seq_len,  # Truncate all sentences.
        )
        
        tokenized_sentences.append(tokenized_sentence)

    return tokenized_sentences

def create_attention_masks(tokenized_and_padded_sentences):
    attention_masks = []

    for sentence in tokenized_and_padded_sentences:
        att_mask = [int(token_id > 0) for token_id in sentence]
        attention_masks.append(att_mask)

    return np.asarray(attention_masks)

def create_dataset(data_tuple, epochs=1, batch_size=32, buffer_size=10000, train=True):
    dataset = tf.data.Dataset.from_tensor_slices(data_tuple)
    if train:
        dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.repeat(epochs)
    dataset = dataset.batch(batch_size)
    if train:
        dataset = dataset.prefetch(1)
    
    return dataset

class BertClassifier(tf.keras.Model):    
    def __init__(self, bert: TFBertModel, num_classes: int):
        super().__init__()
        self.bert = bert
        self.classifier = Dense(num_classes, activation='sigmoid')
        
    @tf.function
    def call(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
        outputs = self.bert(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask)
        cls_output = outputs[1]
        cls_output = self.classifier(cls_output)
                
        return cls_output

@tf.function
def train_step(model, token_ids, masks, labels):
    labels = tf.dtypes.cast(labels, tf.float32)

    with tf.GradientTape() as tape:
        predictions = model(token_ids, attention_mask=masks)
        loss = loss_object(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables), 1.0)

    train_loss(loss)

    for i, auc in enumerate(train_auc_metrics):
        auc.update_state(labels[:,i], predictions[:,i])
        
@tf.function
def validation_step(model, token_ids, masks, labels):
    labels = tf.dtypes.cast(labels, tf.float32)

    predictions = model(token_ids, attention_mask=masks, training=False)
    v_loss = loss_object(labels, predictions)

    validation_loss(v_loss)
    for i, auc in enumerate(validation_auc_metrics):
        auc.update_state(labels[:,i], predictions[:,i])
                                              
def train(model, train_dataset, val_dataset, train_steps_per_epoch, val_steps_per_epoch, epochs):
    for epoch in range(epochs):
        print('=' * 50, f"EPOCH {epoch}", '=' * 50)

        start = time.time()

        for i, (token_ids, masks, labels) in enumerate(train_dataset, train_steps_per_epoch):
            train_step(model, token_ids, masks, labels)
            if i % 1000 == 0:
                print(f'\nTrain Step: {i}, Loss: {train_loss.result()}')
                for i, label_name in enumerate(label_cols):
                    print(f"{label_name} roc_auc {train_auc_metrics[i].result()}")
                    train_auc_metrics[i].reset_states()
        
        for i, (token_ids, masks, labels) in enumerate(val_dataset, val_steps_per_epoch):
            validation_step(model, token_ids, masks, labels)

        print(f'\nEpoch {epoch+1}, Validation Loss: {validation_loss.result()}, Time: {time.time()-start}\n')

        for i, label_name in enumerate(label_cols):
            print(f"{label_name} roc_auc {validation_auc_metrics[i].result()}")
            validation_auc_metrics[i].reset_states()

        print('\n')


if __name__ == "__main__":
    # Data path
    dataset_directory = '../input/jigsaw-toxic-comment-classification-challenge'
    train_path = 'data/train.csv'
    test_path = 'data/test.csv'
    test_labels_path = 'data/test_labels.csv'
    subm_path = 'data/sample_submission.csv'


    # set data
    label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    df_test_labels = pd.read_csv(test_labels_path)
    df_test_labels = df_test_labels.set_index('id')

    print(df_train.head())

    # set model
    bert_model_name = 'bert-base-uncased'

    tokenizer = BertTokenizer.from_pretrained(bert_model_name, do_lower_case=True)
    MAX_LEN = 128

    input_ids = tokenize_sentences(df_train['comment_text'], tokenizer, MAX_LEN)
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", value=0, truncating="post", padding="post")
    attention_masks = create_attention_masks(input_ids)

    # set inputs
    labels =  df_train[label_cols].values

    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, random_state=0, test_size=0.1)
    train_masks, validation_masks, _, _ = train_test_split(attention_masks, labels, random_state=0, test_size=0.1)

    train_size = len(train_inputs)
    validation_size = len(validation_inputs)

    # set dataset
    BATCH_SIZE = 8
    NR_EPOCHS = 1

    train_dataset = create_dataset((train_inputs, train_masks, train_labels), epochs=NR_EPOCHS, batch_size=BATCH_SIZE)
    validation_dataset = create_dataset((validation_inputs, validation_masks, validation_labels), epochs=NR_EPOCHS, batch_size=BATCH_SIZE)

    # create model
    model = BertClassifier(TFBertModel.from_pretrained(bert_model_name), len(label_cols))

    # training Loop
    steps_per_epoch = train_size // BATCH_SIZE
    validation_steps = validation_size // BATCH_SIZE

    # | Loss Function
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    validation_loss = tf.keras.metrics.Mean(name='test_loss')

    # | Optimizer (with 1-cycle-policy)
    warmup_steps = steps_per_epoch // 3
    total_steps = steps_per_epoch * NR_EPOCHS - warmup_steps
    optimizer = create_optimizer(init_lr=2e-5, num_train_steps=total_steps, num_warmup_steps=warmup_steps)

    # | Metrics
    train_auc_metrics = [tf.keras.metrics.AUC() for i in range(len(label_cols))]
    validation_auc_metrics = [tf.keras.metrics.AUC() for i in range(len(label_cols))]

    # start training
    train(model, train_dataset, validation_dataset, train_steps_per_epoch=steps_per_epoch, val_steps_per_epoch=validation_steps, epochs=NR_EPOCHS)





    