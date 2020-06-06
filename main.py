# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 12:41:15 2019

@author: Mariano Leonel Acosta
"""
import numpy as np
import pandas as pd
import utils as U
import pickle
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Average, Concatenate, Conv1D, MaxPooling1D, Flatten, LSTM, Embedding
from sklearn.metrics import balanced_accuracy_score
import keras
# -----------------------------------------------------------------------------------------------------------------------------
#import data

data = pd.read_csv('train.csv')
class_dic = U.label_to_integer(
    list(set(data['category'].to_list())))  # create a class dictionary
c_count = data.category.value_counts().sort_values(
    ascending=False)  # class count (for visualization)

# load tokenizer

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


d_train, d_validation, _, _ = train_test_split(
    data, data['category'], test_size=0.05, stratify=data['category'])


# create smaller training sets

d_tn1, d_tn3, _, _ = train_test_split(
    d_train, d_train['category'], test_size=0.33, stratify=d_train['category'])


d_tn1, d_tn2, _, _ = train_test_split(
    d_tn1, d_tn1['category'], test_size=0.5, stratify=d_tn1['category'])


# ---------------------------------------------------------------------------------------------------------------------------------------
# First model

# oversampling

lst = [d_tn1]

for _, group in d_tn1.groupby('category'):
    if len(group) < 3000:
        lst.append(group.sample(n=3000, replace=True))

frame_new = pd.concat(lst)
frame_new = frame_new.groupby('category').head(3000)

# create sequences
num_words = 30000
with open('tokenizer.pickle', 'rb') as handle:
    tk_30 = pickle.load(handle)

tk_30.word_index = {e: i for e, i in tk_30.word_index.items(
) if i <= num_words}  # <= because tokenizer is 1 indexed
tk_30.word_index[tk_30.oov_token] = num_words + 1
titles = frame_new['title'].tolist()
sequences = tk_30.texts_to_sequences(titles)


max_length = 15

sequences = keras.preprocessing.sequence.pad_sequences(
    sequences, maxlen=max_length, padding='post', truncating='post')

classes = frame_new['category']
labels = []

for cat in classes:
    labels.append(class_dic[cat])


skf_1 = StratifiedKFold(n_splits=5, shuffle=False)

# create K FOLD index for cross-validation
trn_1_inx = []
tst_1_inx = []

for train_index, test_index in skf_1.split(sequences, labels):

    trn_1_inx.append(train_index)
    tst_1_inx.append(test_index)


batch_size = 128

model_1 = Sequential()

model_1.add(Embedding(30000 + 2, 32, input_length=max_length))
model_1.add(LSTM(128, return_sequences=False))

model_1.add(Dense(512, activation='relu'))
model_1.add(Dense(1588, activation='softmax'))

opt = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.99, beta_2=0.999)

model_1.compile(loss='sparse_categorical_crossentropy',
                optimizer=opt, metrics=['accuracy'])

# training
balanced = []

for inx in range(2):
    x_train = sequences[np.array(trn_1_inx[inx]), :]
    x_test = sequences[np.array(tst_1_inx[inx]), :]
    y_train = np.asarray(labels)[np.array(trn_1_inx[inx])]

#    class_weights = class_weight.compute_class_weight('balanced',list(set(y_train)),y_train)

    model_1.fit(
        x_train,
        y_train,
        epochs=1,
        shuffle=True,
        verbose=1,
        batch_size=batch_size)

    # balanced accuracy
    y_test = np.asarray(labels)[np.array(tst_1_inx[inx])]

    y_p = model_1.predict(x_test, batch_size=128, verbose=1)
    y_p = np.argmax(y_p, axis=1)

    balanced.append(balanced_accuracy_score(y_test, y_p))

    print('\nbalanced accuracy: %f\n' % (balanced[inx]))

model_1.save('model_01.h5')

# ---------------------------------------------------------------------------------------------------------------------------------------
# Second model

# oversampling
lst = [d_tn2]

for _, group in d_tn2.groupby('category'):
    if len(group) < 3000:
        lst.append(group.sample(n=3000, replace=True))

frame_new = pd.concat(lst)
frame_new = frame_new.groupby('category').head(3000)

# create sequences
num_words = 30000
with open('tokenizer.pickle', 'rb') as handle:
    tk_30 = pickle.load(handle)

tk_30.word_index = {e: i for e, i in tk_30.word_index.items(
) if i <= num_words}  # <= because tokenizer is 1 indexed
tk_30.word_index[tk_30.oov_token] = num_words + 1
titles = frame_new['title'].tolist()
sequences = tk_30.texts_to_sequences(titles)


# pad sequences

max_length = 15

sequences = keras.preprocessing.sequence.pad_sequences(
    sequences, maxlen=max_length, padding='post', truncating='post')

classes = frame_new['category']
labels = []

for cat in classes:
    labels.append(class_dic[cat])


skf_2 = StratifiedKFold(n_splits=5, shuffle=False)

# create K FOLD index for cross-validation
trn_2_inx = []
tst_2_inx = []

for train_index, test_index in skf_2.split(sequences, labels):

    trn_2_inx.append(train_index)
    tst_2_inx.append(test_index)


batch_size = 128

model_2 = Sequential()

model_2.add(Embedding(30000 + 2, 50, input_length=max_length))

model_2.add(LSTM(128, return_sequences=False))

model_2.add(Dense(1588, activation='softmax'))

opt = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.99, beta_2=0.999)

model_2.compile(loss='sparse_categorical_crossentropy',
                optimizer=opt, metrics=['accuracy'])

# training
balanced = []

for inx in range(5):
    x_train = sequences[np.array(trn_2_inx[inx]), :]
    x_test = sequences[np.array(tst_2_inx[inx]), :]
    y_train = np.asarray(labels)[np.array(trn_2_inx[inx])]

    class_weights = class_weight.compute_class_weight(
        'balanced', list(set(y_train)), y_train)

    model_2.fit(
        x_train,
        y_train,
        epochs=1,
        shuffle=True,
        verbose=1,
        batch_size=batch_size,
        class_weight=class_weights)

    # balanced accuracy
    y_test = np.asarray(labels)[np.array(tst_2_inx[inx])]

    y_p = model_2.predict(x_test, batch_size=128, verbose=1)
    y_p = np.argmax(y_p, axis=1)

    balanced.append(balanced_accuracy_score(y_test, y_p))

    print('\nbalanced accuracy: %f\n' % (balanced[inx]))

# model_2.load_weights('model_02.h5')

model_2.save('model_02.h5')
# ---------------------------------------------------------------------------------------------------------------------------------------
# Third model

# oversampling
lst = [d_tn3]

for _, group in d_tn3.groupby('category'):
    if len(group) < 2000:
        lst.append(group.sample(n=2000, replace=True))

frame_new = pd.concat(lst)
frame_new = frame_new.groupby('category').head(2000)

# create sequences
num_words = 60000
with open('tokenizer.pickle', 'rb') as handle:
    tk_60 = pickle.load(handle)

tk_60.word_index = {e: i for e, i in tk_60.word_index.items(
) if i <= num_words}  # <= because tokenizer is 1 indexed
tk_60.word_index[tk_60.oov_token] = num_words + 1
titles = frame_new['title'].tolist()
sequences = tk_60.texts_to_sequences(titles)


# pad sequences

max_length = 15

sequences = keras.preprocessing.sequence.pad_sequences(
    sequences, maxlen=max_length, padding='post', truncating='post')

classes = frame_new['category']
labels = []

for cat in classes:
    labels.append(class_dic[cat])


skf_3 = StratifiedKFold(n_splits=5, shuffle=False)

# create K FOLD index for cross-validation
trn_3_inx = []
tst_3_inx = []

for train_index, test_index in skf_3.split(sequences, labels):

    trn_3_inx.append(train_index)
    tst_3_inx.append(test_index)


batch_size = 128

model_3 = Sequential()

model_3.add(Embedding(60000 + 2, 50, input_length=max_length))

model_3.add(LSTM(160, return_sequences=False))

model_3.add(Dense(1588, activation='softmax'))

opt = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.99, beta_2=0.999)

model_3.compile(loss='sparse_categorical_crossentropy',
                optimizer=opt, metrics=['accuracy'])

# training
balanced = []

for inx in range(2):
    x_train = sequences[np.array(trn_3_inx[inx]), :]
    x_test = sequences[np.array(tst_3_inx[inx]), :]
    y_train = np.asarray(labels)[np.array(trn_3_inx[inx])]

    class_weights = class_weight.compute_class_weight(
        'balanced', list(set(y_train)), y_train)

    model_3.fit(
        x_train,
        y_train,
        epochs=1,
        shuffle=True,
        verbose=1,
        batch_size=batch_size,
        class_weight=class_weights)

    # balanced accuracy
    y_test = np.asarray(labels)[np.array(tst_3_inx[inx])]

    y_p = model_3.predict(x_test, batch_size=128, verbose=1)
    y_p = np.argmax(y_p, axis=1)

    balanced.append(balanced_accuracy_score(y_test, y_p))

    print('\nbalanced accuracy: %f\n' % (balanced[inx]))

model_3.save('model_03.h5')
# model_3.load_weights('model_03.h5')

# -------------------------------------------------------------------------------------------------------------------------
# FEWER CLASSES Fourth Model
# can also index sheet by name or fetch all sheet
df = pd.read_excel('CLASSES.xlsX')

keys = df['ORIGINAL'].tolist()
new_keys = df['NEW'].tolist()

d_new = d_tn1['category'].replace(keys, new_keys, inplace=True)

new_class_dic = U.label_to_integer(list(set(d_new['category'].to_list())))

c_count = d_new.category.value_counts().sort_values(ascending=False)

# oversampling
lst = [d_new]

for _, group in d_new.groupby('category'):
    if len(group) < 5000:
        lst.append(group.sample(n=5000, replace=True))

frame_new = pd.concat(lst)

# create sequences
num_words = 60000
with open('tokenizer.pickle', 'rb') as handle:
    tk_60 = pickle.load(handle)

tk_60.word_index = {e: i for e, i in tk_60.word_index.items(
) if i <= num_words}  # <= because tokenizer is 1 indexed
tk_60.word_index[tk_60.oov_token] = num_words + 1
titles = frame_new['title'].tolist()
sequences = tk_60.texts_to_sequences(titles)


# pad sequences

max_length = 15

sequences = keras.preprocessing.sequence.pad_sequences(
    sequences, maxlen=max_length, padding='post', truncating='post')

classes = frame_new['category']
labels = []

for cat in classes:
    labels.append(new_class_dic[cat])


skf_3 = StratifiedKFold(n_splits=5, shuffle=False)

# create K FOLD index for cross-validation
trn_3_inx = []
tst_3_inx = []

for train_index, test_index in skf_3.split(sequences, labels):

    trn_3_inx.append(train_index)
    tst_3_inx.append(test_index)


batch_size = 128

model_4 = Sequential()

model_4.add(Embedding(60000 + 2, 50, input_length=max_length))

model_4.add(LSTM(128, return_sequences=False))

model_4.add(Dense(445, activation='softmax'))

opt = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.99, beta_2=0.999)

model_4.compile(loss='sparse_categorical_crossentropy',
                optimizer=opt, metrics=['accuracy'])

# training
balanced = []

for inx in range(0, 2):
    x_train = sequences[np.array(trn_3_inx[inx]), :]
    x_test = sequences[np.array(tst_3_inx[inx]), :]
    y_train = np.asarray(labels)[np.array(trn_3_inx[inx])]

    class_weights = class_weight.compute_class_weight(
        'balanced', list(set(y_train)), y_train)

    model_4.fit(
        x_train,
        y_train,
        epochs=1,
        shuffle=True,
        verbose=1,
        batch_size=batch_size,
        class_weight=class_weights)

    # balanced accuracy
    y_test = np.asarray(labels)[np.array(tst_3_inx[inx])]

    y_p = model_4.predict(x_test, batch_size=128, verbose=1)
    y_p = np.argmax(y_p, axis=1)

    balanced = balanced_accuracy_score(y_test, y_p)

    print('\nbalanced accuracy: %f\n' % (balanced))

model_4.save('model_04.h5')
# model_4.load_weights('model_04.h5')

# -------------------------------------------------------------------------------------------------------------------------------
# Fifth Model

batch_size = 128

model_5 = Sequential()

model_5.add(Embedding(30000 + 2, 50, input_length=max_length))

model_5.add(LSTM(128, return_sequences=False))


model_5.add(Dense(1588, activation='softmax'))

opt = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.99, beta_2=0.999)

model_5.compile(loss='sparse_categorical_crossentropy',
                optimizer=opt, metrics=['accuracy'])

model_5.load_weights('model_05.h5')
# -------------------------------------------------------------------------------------------------------------------------------------------
# Conv NET 1
model_conv_1 = Sequential()

model_conv_1.add(Embedding(30000 + 2, 32, input_length=max_length))
# model.add(Dropout(0.2))
model_conv_1.add(Conv1D(64, 4, activation='relu'))

model_conv_1.add(MaxPooling1D(4))
model_conv_1.add(Flatten())
model_conv_1.add(Dense(1024, activation='relu'))
model_conv_1.add(Dense(1588, activation='softmax'))

opt = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.99, beta_2=0.999)

model_conv_1.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy'])

# training
balanced = []

for inx in range(2, 3):
    x_train = sequences[np.array(trn_1_inx[inx]), :]
    x_test = sequences[np.array(tst_1_inx[inx]), :]
    y_train = np.asarray(labels)[np.array(trn_1_inx[inx])]

#    class_weights = class_weight.compute_class_weight('balanced',list(set(y_train)),y_train)

    model_conv_1.fit(
        x_train,
        y_train,
        epochs=1,
        shuffle=True,
        verbose=1,
        batch_size=batch_size)

    # balanced accuracy
    y_test = np.asarray(labels)[np.array(tst_1_inx[inx])]

    y_p = model_conv_1.predict(x_test, batch_size=128, verbose=1)
    y_p = np.argmax(y_p, axis=1)

    balanced = balanced_accuracy_score(y_test, y_p)

    print('\nbalanced accuracy: %f\n' % (balanced))

model_conv_1.save('model_conv_01.h5')


# ------------------------------------------------------------------------------------------------------------------------------------------
# Conv NET + LSTM (USE THIRD DATASET)


# oversampling
lst = [d_tn3]

for _, group in d_tn3.groupby('category'):
    if len(group) < 4000:
        lst.append(group.sample(n=4000, replace=True))

frame_new = pd.concat(lst)
frame_new = frame_new.groupby('category').head(4000)

# create sequences
num_words = 60000
with open('tokenizer.pickle', 'rb') as handle:
    tk_60 = pickle.load(handle)

tk_60.word_index = {e: i for e, i in tk_60.word_index.items(
) if i <= num_words}  # <= because tokenizer is 1 indexed
tk_60.word_index[tk_60.oov_token] = num_words + 1
titles = frame_new['title'].tolist()
sequences = tk_60.texts_to_sequences(titles)


# pad sequences

max_length = 15

sequences = keras.preprocessing.sequence.pad_sequences(
    sequences, maxlen=max_length, padding='post', truncating='post')

classes = frame_new['category']
labels = []

for cat in classes:
    labels.append(class_dic[cat])


skf_3 = StratifiedKFold(n_splits=5, shuffle=False)

# create K FOLD index for cross-validation
trn_3_inx = []
tst_3_inx = []

for train_index, test_index in skf_3.split(sequences, labels):

    trn_3_inx.append(train_index)
    tst_3_inx.append(test_index)


model_conv_2 = Sequential()

model_conv_2.add(Embedding(60000 + 2, 40, input_length=max_length))
# model.add(Dropout(0.2))
model_conv_2.add(Conv1D(64, 4, activation='relu'))

model_conv_2.add(MaxPooling1D(2))
model_conv_2.add(LSTM(128, return_sequences=False))

model_conv_2.add(Dense(1588, activation='softmax'))

opt = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.99, beta_2=0.999)

model_conv_2.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy'])

# training
balanced = []

for inx in range(2, 5):
    x_train = sequences[np.array(trn_3_inx[inx]), :]
    x_test = sequences[np.array(tst_3_inx[inx]), :]
    y_train = np.asarray(labels)[np.array(trn_3_inx[inx])]

    class_weights = class_weight.compute_class_weight(
        'balanced', list(set(y_train)), y_train)

    model_conv_2.fit(
        x_train,
        y_train,
        epochs=1,
        shuffle=True,
        verbose=1,
        batch_size=batch_size,
        class_weight=class_weights)

    # balanced accuracy
    y_test = np.asarray(labels)[np.array(tst_3_inx[inx])]

    y_p = model_conv_2.predict(x_test, batch_size=128, verbose=1)
    y_p = np.argmax(y_p, axis=1)

    balanced = balanced_accuracy_score(y_test, y_p)

    print('\nbalanced accuracy: %f\n' % (balanced))

model_conv_2.save('model_conv_02.h5')


# ------------------------------------------------------------------------------------------------------------------------------------------
# Deeper Conv NET


model_conv_3 = Sequential()

model_conv_3.add(Embedding(60000 + 2, 50, input_length=max_length))
# model.add(Dropout(0.2))
model_conv_3.add(Conv1D(128, 3, activation='relu'))

model_conv_3.add(MaxPooling1D(2))

model_conv_3.add(Conv1D(256, 3, activation='relu'))

# model_conv_3.add(MaxPooling1D(2))


model_conv_3.add(Flatten())
model_conv_3.add(Dense(1024, activation='relu'))
model_conv_3.add(Dense(1588, activation='softmax'))

opt = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

model_conv_3.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy'])

# training
balanced = []

for inx in range(5):
    x_train = sequences[np.array(trn_3_inx[inx]), :]
    x_test = sequences[np.array(tst_3_inx[inx]), :]
    y_train = np.asarray(labels)[np.array(trn_3_inx[inx])]

    class_weights = class_weight.compute_class_weight(
        'balanced', list(set(y_train)), y_train)

    model_conv_3.fit(
        x_train,
        y_train,
        epochs=1,
        shuffle=True,
        verbose=1,
        batch_size=batch_size,
        class_weight=class_weights)

    # balanced accuracy
    y_test = np.asarray(labels)[np.array(tst_3_inx[inx])]

    y_p = model_conv_3.predict(x_test, batch_size=128, verbose=1)
    y_p = np.argmax(y_p, axis=1)

    balanced = balanced_accuracy_score(y_test, y_p)

    print('\nbalanced accuracy: %f\n' % (balanced))

model_conv_3.save('model_conv_03.h5')


# ------------------------------------------------------------------------------------------------------------------------------------------

# print results

max_length = 15

data_test = pd.read_csv(
    'F:\\Programacion\\Python\\Deep Learning\\ML Data Challenge\\test.csv')

# reverse the dictionary
class_dic_reversed = {v: k for k, v in class_dic.items()}

seq_test_30 = tk_30.texts_to_sequences(data_test['title'].tolist())
seq_test_30 = keras.preprocessing.sequence.pad_sequences(
    seq_test_30, maxlen=max_length, padding='post', truncating='post')

seq_test_60 = tk_60.texts_to_sequences(data_test['title'].tolist())
seq_test_60 = keras.preprocessing.sequence.pad_sequences(
    seq_test_60, maxlen=max_length, padding='post', truncating='post')


y_pred_2 = model_5.predict(seq_test_30, verbose=1)
y_pred_3 = model_2.predict(seq_test_30, verbose=1)  # new
y_pred_4 = model_3.predict(seq_test_60, verbose=1)  # new


y_pred = (y_pred_2 + y_pred_3 + y_pred_4) / 3

y_pred = np.argmax(y_pred, axis=1)

categories = [class_dic_reversed[x] for x in y_pred.tolist()]

data_test['category'] = categories

output = data_test.drop(['title', 'language'], axis=1)

output.to_csv(
    r'F:\\Programacion\\Python\\Deep Learning\\ML Data Challenge\\ENSEMBLE\\NEW TRY\result.csv',
    index=False)


# -------------------------------------------------------------------------------------------------------------------------------

members = [model_5, model_2, model_3]

for i in range(len(members)):
    model = members[i]

    for layer in model.layers:
        # make not trainable
        layer.trainable = False

for layer in model_4.layers:
    layer.trainable = False

    # define multi-headed input


ensemble_visible = [model.input for model in members] + [model_4.input]
# concatenate merge output from each model
ensemble_outputs = [model.output for model in members]
merge = Average()(ensemble_outputs)
merge2 = Concatenate()([merge, model_4.output])
hidden = Dense(3200, activation='relu')(merge2)
output = Dense(1588, activation='softmax')(hidden)
opt = tf.keras.optimizers.Adam(lr=1e-3, beta_1=0.99, beta_2=0.999)
model_s4 = Model(inputs=ensemble_visible, outputs=output)


# compile
model_s4.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy'])
model_s4.load_weights('model_stack_04.h5')


# FINAL ENSEMBLE----------------------------------------------------------
#y_p_1 = model_s2.predict([seq_test_30,seq_test_30,seq_test_60,seq_test_60],batch_size=128, verbose=1)
#y_p_2 = model_s3.predict([seq_test_30,seq_test_30,seq_test_60,seq_test_60],batch_size=128, verbose=1)
y_p_3 = model_s4.predict([seq_test_30,
                          seq_test_30,
                          seq_test_60,
                          seq_test_60],
                         batch_size=128,
                         verbose=1)

y_pred_2 = model_5.predict(seq_test_30, verbose=1)
y_pred_3 = model_2.predict(seq_test_30, verbose=1)
y_pred_4 = model_3.predict(seq_test_60, verbose=1)


y_pred_5 = model_1.predict(seq_test_30, verbose=1)
y_pred_6 = model_conv_1.predict(seq_test_30, verbose=1)
y_pred_7 = model_conv_2.predict(seq_test_60, verbose=1)
y_pred_8 = model_conv_3.predict(seq_test_60, verbose=1)

# weighted average
y_p = (y_p_3 + y_pred_7 + y_pred_8) * .55 + (y_pred_2 +
                                             y_pred_3 + y_pred_4 + y_pred_5 + y_pred_6) * .45

#y_p = (y_p_1 +y_p_2 +y_p_3+y_pred_7+y_pred_8)*.55 + (y_pred_2+y_pred_3+y_pred_4+y_pred_5+y_pred_6)*.45

y_pred = np.argmax(y_p, axis=1)

categories = [class_dic_reversed[x] for x in y_pred.tolist()]

data_test['category'] = categories

output = data_test.drop(['title', 'language'], axis=1)

output.to_csv(r'result.csv', index=False)
