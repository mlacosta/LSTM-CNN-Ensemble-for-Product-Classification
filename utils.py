# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 15:48:09 2019

@author: Mariano Leonel Acosta

A small library that I created for the ML data challenge
"""
import pandas as pd
import numpy as np
import random
from keras import backend as K


def unique_chars(dataframe):

    dataList = dataframe['title'].to_list()
    chars = []
    seen = set()

    for data in dataList:
        char_temp = list(set(data.lower()))
        for c in char_temp:
            if c not in seen:
                chars.append(c)
                seen.add(c)

    print(sorted(chars))
    vocab_size = len(chars)
    print('There are %d unique characters in your data.' % (vocab_size))
    return chars


def remove_chars(dataframe, chars):

    dataList = dataframe['title'].to_list()
    newTitles = []

    for title in dataList:

        new = title
        for c in chars:
            new = new.replace(c, '')

        new = new.lower()
        newTitles.append(new)

    dataframe['title'] = pd.Series(newTitles).values
    return dataframe


def replace_chars(dataframe, chars, letter):

    dataList = dataframe['title'].to_list()
    newTitles = []

    for title in dataList:

        new = title

        for c in chars:
            new = new.replace(c, letter)

        new = new.lower()
        newTitles.append(new)

    dataframe['title'] = pd.Series(newTitles).values
    return dataframe


def remove_spaces(dataframe):

    dataList = dataframe['title'].to_list()
    newTitles = []

    for title in dataList:
        new = title
        words = title_to_words(new)
        new = ' '.join(words)
        newTitles.append(new)

    dataframe['title'] = pd.Series(newTitles).values
    return dataframe


def create_dictionary(dataframe):

    dataList = dataframe['title'].to_list()
    dictionary = {}
    length = len(dataList)
    counter = 0

    for title in dataList:
        words = []
        words = title.split(' ')

        if ' ' in words:
            words = words.remove(' ')

        for w in words:
            if w in dictionary:
                dictionary[w] = dictionary[w] + 1
            else:
                dictionary[w] = 1

        counter += 1

        if counter % 1000 == 0:
            print('%%%3.2f processed\n' % (counter / length * 100))

    return dictionary


def create_lookup(words):

    forward = {}
    backward = {}
    length = len(words)

    for inx in range(length):
        forward[words[inx]] = inx
        backward[inx] = words[inx]

    return forward, backward


def get_integer(forward, words):

    # input: a list of words
    # output: list of integers

    integers = []

    for w in words:
        try:
            integers.append(forward[w])
        except BaseException:
            integers.append(forward['UNK'])

    return integers


def get_words(backward, integers):

    # input: a list of integer
    # output: list of words

    words = []

    for i in integers:
        words.append(backward[i])

    return words


def title_to_words(title):
    # input: a string (title)
    # output: list of words in title
    words = []
    words = title.split(' ')

    while("" in words):
        words.remove("")

    return words


def create_word_list(sorted_dict, num_of_words=10000):

    newWords = []
    wordFreq = []

    summ = 0

    for inx in range(num_of_words - 1, len(sorted_dict)):
        summ += sorted_dict[inx][1]

    newWords.append('UNK')
    wordFreq.append(summ)

    for inx in range(num_of_words - 1):
        newWords.append(sorted_dict[inx][0])
        wordFreq.append(sorted_dict[inx][1])

    return newWords, wordFreq


def label_to_integer(class_list):

    class_dic = {}
    inx = 0

    for cat in class_list:
        class_dic[cat] = inx
        inx += 1

    return class_dic


def int_to_one_hot(integer, size):

    one_hot = [0] * size
    one_hot[integer] = 1
    one_hot = np.array(one_hot, dtype=int)

    return np.reshape(one_hot, (1, size))


def create_one_hot_labels(classes):

    unique = np.array(pd.unique(classes), dtype=str)
    class_list = unique.tolist()
    class_dic = label_to_integer(class_list)
    size = len(class_list)
    labels = np.empty((1, size))

    for cat in classes:
        integer = class_dic[cat]
        np.concatenate((labels, int_to_one_hot(integer, size)), axis=0)

    labels = np.delete(labels, (0), axis=0)

    return labels


def embedded_vector(integer, embedded_matrix):
    temp = embedded_matrix[integer, :]
    return np.reshape(temp, (1, embedded_matrix.shape[1]))


def pad_sequence(sequence, length):

    l = len(sequence)
    new_sequence = []
    new_sequence += sequence

    for inx in range(length - l):
        new_sequence += [0]

    return new_sequence


def dataset_to_train(dataset, max_length, forward, class_dic):

    titles = dataset['title']
    classes = dataset['category']
    x_data = np.empty((1, max_length), dtype=int)
    y_data = np.empty((1, 1), dtype=int)
    counter = 0
    total = len(dataset)

    for title in titles:

        words = title_to_words(title)
        integers = get_integer(forward, words)
        length = len(integers)

        if length >= max_length:
            integers = integers[0:max_length]
        else:
            integers = pad_sequence(integers, max_length)

        x_data = np.append(
            x_data, (np.reshape(
                np.asarray(integers), (1, max_length))), axis=0)
        counter += 1

        if counter % 100 == 0:
            print('%%%3.3f\n' % (counter / total * 100))

    x_data = np.delete(x_data, (0), axis=0)
    counter = 0

    for category in classes:

        y_data = np.append(y_data, np.reshape(
            np.asarray([class_dic[category]]), (1, 1)), axis=0)
        counter += 1

        if counter % 100 == 0:
            print('%%%3.3f\n' % (counter / total * 100))
    y_data = np.delete(y_data, (0), axis=0)

    return x_data, y_data


def get_input(path):

    x_data = np.load(path)
    return x_data


def pre_process(x, y, batch_size):

    l = x.shape[0]
    inx = random.randint(0, l - batch_size)

    return x[inx:(inx + batch_size), :], y[inx:(inx + batch_size), :]


def generator(x_path, y_path, batch_size):

    while True:
        x = get_input(x_path)
        y = get_input(y_path)
        (batch_x, batch_y) = pre_process(x, y, batch_size)

    yield(batch_x, batch_y)


def recall_m(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):

    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))
