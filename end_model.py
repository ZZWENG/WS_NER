from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib.layers.crf import CRF
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

MAX_LEN = 7  # Max length of review (in words)
EMBEDDING = 20
DROPOUT = 0.2

def train_end_model(train_df, dev_df, test_df, n_epochs, batch_size=16):
    train_sentences = []
    words = set()
    for i, row in train_df.iterrows():
        sent = list(zip(row['product'].split(), row['label'].split()))
        train_sentences += [sent]
        words.update(row['product'].split())

    dev_sentences = []
    for i, row in dev_df.iterrows():
        sent = list(zip(row['product'].split(), row['label'].split()))
        dev_sentences += [sent]
        words.update(row['product'].split())

    test_sentences = []
    for i, row in test_df.iterrows():
        sent = list(zip(row['product'].split(), row['label'].split()))
        test_sentences += [sent]
        words.update(row['product'].split())

    words = list(words)
    tags = ['other', 'category', 'modelname', 'brand']
    n_tags = len(tags)
    n_words = len(words)  # vocabulary size

    # Vocabulary Key:word -> Value:token_index
    # The first 2 entries are reserved for PAD and UNK
    word2idx = {w: i + 2 for i, w in enumerate(words)}
    word2idx["UNK"] = 1 # Unknown words
    word2idx["PAD"] = 0 # Padding

    # Vocabulary Key:token_index -> Value:word
    idx2word = {i: w for w, i in word2idx.items()}

    # Vocabulary Key:Label/Tag -> Value:tag_index
    # The first entry is reserved for PAD
    tag2idx = {t: i+1 for i, t in enumerate(tags)}
    tag2idx["PAD"] = 0

    # Vocabulary Key:tag_index -> Value:Label/Tag
    idx2tag = {i: w for w, i in tag2idx.items()}

    # Convert each sentence from list of Token to list of word_index
    X_train = [[word2idx[w[0]] for w in s] for s in train_sentences]
    X_train = pad_sequences(maxlen=MAX_LEN, sequences=X_train, padding="post", value=word2idx["PAD"])
    y_train = [[tag2idx[w[1]] for w in s] for s in train_sentences]
    y_train = pad_sequences(maxlen=MAX_LEN, sequences=y_train, padding="post", value=tag2idx["PAD"])

    X_dev = [[word2idx[w[0]] for w in s] for s in dev_sentences]
    X_dev = pad_sequences(maxlen=MAX_LEN, sequences=X_dev, padding="post", value=word2idx["PAD"])
    y_dev = [[tag2idx[w[1]] for w in s] for s in dev_sentences]
    y_dev = pad_sequences(maxlen=MAX_LEN, sequences=y_dev, padding="post", value=tag2idx["PAD"])

    X_test = [[word2idx[w[0]] for w in s] for s in test_sentences]
    X_test = pad_sequences(maxlen=MAX_LEN, sequences=X_test, padding="post", value=word2idx["PAD"])
    y_test = [[tag2idx[w[1]] for w in s] for s in test_sentences]
    y_test = pad_sequences(maxlen=MAX_LEN, sequences=y_test, padding="post", value=tag2idx["PAD"])

    # One-Hot encode
    y_train = [to_categorical(i, num_classes=n_tags+1) for i in y_train]  # n_tags+1(PAD)
    y_dev = [to_categorical(i, num_classes=n_tags+1) for i in y_dev]  # n_tags+1(PAD)
    y_test = [to_categorical(i, num_classes=n_tags+1) for i in y_test]  # n_tags+1(PAD)

    # Model definition
    input = Input(shape=(MAX_LEN,))
    model = Embedding(input_dim=n_words+2, output_dim=EMBEDDING, # n_words + 2 (PAD & UNK)
                      input_length=MAX_LEN, mask_zero=True)(input)  # default: 20-dim embedding
    model = Bidirectional(LSTM(units=50, return_sequences=True,
                               recurrent_dropout=DROPOUT))(model)  # variational biLSTM
    model = TimeDistributed(Dense(50, activation="relu"))(model)  # a dense layer as suggested by neuralNer
    crf = CRF(n_tags+1)  # CRF layer, n_tags+1(PAD)
    out = crf(model)  # output

    model = Model(input, out)
    model.compile(optimizer="adam", loss=crf.loss_function, metrics=[crf.accuracy])

    history = model.fit(X_train, np.array(y_train), batch_size=batch_size, epochs=n_epochs,
                        validation_data=(X_dev, np.array(y_dev)), verbose=2)
    # Eval
    pred_cat = model.predict(X_test)
    pred = np.argmax(pred_cat, axis=-1)
    y_test_true = np.argmax(y_test, -1)

    pred_tag = [[idx2tag[i] for i in row] for row in pred]
    y_test_true_tag = [[idx2tag[i] for i in row] for row in y_test_true]

    return y_test_true_tag, pred_tag