import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

import plotly
import plotly.graph_objs as go
from matplotlib import cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import ast

root = "train_word_emb/"
#read dictionary file
dict_file = open(root + "dict.txt", "r") 
lines = dict_file.readlines()
word_2_idx = {}
for i, d in enumerate(lines):
    word_2_idx = ast.literal_eval(d)
# print(word_2_idx['keycard'])

#read word embedding file
emb_file = open(root + "w_emb.txt", "r")
embs_string = []
full_embs_string = []
lines = emb_file.readlines()

temp = ""
for i, emb in enumerate(lines):
    if "[[" in emb or "[" in emb: #start
        temp = ""
        if "[[" in emb:
            temp += emb[2:]
        elif "[" in emb:
            temp += emb[3:]
    elif "]" in emb:
        if "]]" in emb:
            temp += emb[:len(emb) - 3]
            embs_string.append(temp)
            full_embs_string.append(embs_string)
            embs_string = []
        else:
            temp += emb[:len(emb) - 2]
            embs_string.append(temp)
    else:
        temp += emb

full_embs_number = []
for i in range(len(full_embs_string)):
    tmp_embs_number = []
    for w_emb in full_embs_string[i]:
        temp = w_emb.split(" ")
        temp_list = []
        for num in temp:
            if num == '':
                continue
            if '\n' in num:
                temp_list.append(float(num[:-2]))
            else:
                temp_list.append(float(num))
        
        tmp_embs_number.append(temp_list)
    full_embs_number.append(tmp_embs_number)

print(len(full_embs_number))
# print(len(full_embs_number[0]))
# print(len(full_embs_number[0][0]))

idx_file = open(root + "word.txt", "r")
idxs = []
lines = idx_file.readlines()
for i, sen in enumerate(lines):
    sen = sen[1:-2]
    idxs_string = sen.split()
    idxs.append([int(x[:-1]) for x in idxs_string])

print(len(idxs))

def get_word_to_idx(idx_2_word):
    word2vec = {}
    for w_idxs, w_embs  in zip(idxs, full_embs_number):
        for k, w_idx in enumerate(w_idxs):
            word = idx_2_word[w_idx]
            if word not in word2vec:
                word2vec[word] = w_embs[k]

    return word2vec

idx_2_word = {y:x for x,y in word_2_idx.items()}
word2vec = get_word_to_idx(idx_2_word)
# make sentences
sentences = []
for idx in idxs:
    string = ''
    for i in idx:
        string += idx_2_word[i] + " "
    string = string[:-1]
    sentences.append(string)

# print(word2vec)

# load sentence vectors for earch word
lstm_vector_dict = {}
for key in word_2_idx:
    file_name = key+ "_emb.txt"
    path = root + "/dict_emb/" + file_name
    f = open(path, "r")

    lines = f.readlines()

    temp = ""
    for i, emb in enumerate(lines):
        if "[" in emb: #start
            temp = ""
            temp += emb[1:]
        elif "]" in emb:
            temp += emb[:len(emb) - 2]
            embs_string.append(temp)
        else:
            temp += emb
    
    temp = temp.split(" ")
    temp_list = []
    for i in temp:
        if i == '':
            continue
        if '\n' in i:
            temp_list.append(float(i[:-2]))
        else:
            temp_list.append(float(i))

    lstm_vector_dict[key] = temp_list


import numpy as np
import matplotlib.pyplot as plt
from random import randint
import torch
import torch.nn as nn

def get_batch(sent):
    embed = np.zeros((len(sent[0]), len(sent), 32))
    for i in range(len(sent)):
        for j in range(len(sent[i])):
            embed[j, i, :] = word2vec[sent[i][j]]
    return torch.FloatTensor(embed)

def get_batch2(sent):
    output = np.zeros((len(sent[0]), len(sent), 256))
    for i, word in enumerate(sent[0]):
        vector = lstm_vector_dict[word]
        output[i, 0, :] = vector
    return torch.FloatTensor(output)

def visualize2(sent, tokenize=True):
    sent = sent.split()
    sent = [[word for word in sent]]

    batch = get_batch(sent)

    # enc_lstm = nn.GRU(32, 256, 1, bidirectional=False, dropout=0.0).to('cuda')
    # batch = batch.cuda()
    # output = enc_lstm(batch)[0]

    output = get_batch2(sent)
    output, idxs = torch.max(output, 0)

    idxs = idxs.data.cpu().numpy()
    argmaxs = [np.sum((idxs == k)) for k in range(len(sent[0]))]

    # visualize model
    x = range(len(sent[0]))
    y = [100.0 * n / np.sum(argmaxs) for n in argmaxs]
    print(sum(y))
    plt.xticks(x, sent[0], rotation=45)
    plt.bar(x, y)
    plt.ylabel('The Percentage of Importance (%)')
    plt.xlabel('Words')
    plt.title('Visualization of Words Importance')
    plt.show()

    return output, idxs

idx = randint(0, len(sentences))
# while 'Go to the smallest object' not in sentences[idx]:
#     idx = randint(0, len(sentences))    

_, _ = visualize2(sentences[idx])



