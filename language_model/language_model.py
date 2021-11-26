import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import itertools
import json,os 
import torch.nn.functional as F
from fc import FCNet

def create_glove_embedding_init(idx2word, glove_file):
    word2emb = {}

    with open(glove_file, 'r', encoding='utf-8') as f:
        entries = f.readlines()
    
    emb_dim = len(entries[0].split(' ')) - 2 #eliminate '\n'
    print('embedding dim is %d' % emb_dim)
    weights = np.zeros((len(idx2word), emb_dim), dtype=np.float32)

    for entry in entries:
        vals = entry.split(' ')
        word = vals[0]
        
        vals = list(map(float, vals[1:-1])) #eliminate '\n'
        word2emb[word] = np.array(vals)
    for idx, word in enumerate(idx2word):
        if word not in word2emb:
            continue
        weights[idx] = word2emb[word]
    
    return weights, word2emb

def tokenize(text, word2idx):
    tokens = []
    words = text.split(' ')
    for w in words:
        tokens.append(word2idx[w])
    
    return tokens

def tfidf_from_instructions(dictionary, dataroot='data'):
    inds = [[], []] # rows, cols for uncoalesce sparse matrix
    df = dict()
    N = len(dictionary)
    
    def populate(inds, df, text):
        tokens = tokenize(text, dictionary)
        for t in tokens:
            df[t] = df.get(t, 0) + 1
        combin = list(itertools.combinations(tokens, 2))
        for c in combin:
            if c[0] < N:
                inds[0].append(c[0]); inds[1].append(c[1])
            if c[1] < N:
                inds[0].append(c[1]); inds[1].append(c[0])

    question_path = dataroot + "/instructions_all.json"
    questions = json.load(open(question_path))
    for question in questions:
        populate(inds, df, question['instruction'])

    # TF-IDF
    vals = [1] * len(inds[1])
    for idx, col in enumerate(inds[1]):
        assert df[col] >= 1, 'document frequency should be greater than zero!'
        vals[col] /= df[col]

    # Make stochastic matrix
    def normalize(inds, vals):
        z = dict()
        for row, val in zip(inds[0], vals):
            z[row] = z.get(row, 0) + val
        for idx, row in enumerate(inds[0]):
            vals[idx] /= z[row]
        return vals

    vals = normalize(inds, vals)

    tfidf = torch.sparse.FloatTensor(torch.LongTensor(inds), torch.FloatTensor(vals))
    tfidf = tfidf.coalesce()

    idx2word = []
    temp_dict = {k: v for k, v in sorted(dictionary.items(), key=lambda item: item[1])}
    
    for w in temp_dict:
        idx2word.append(w)   
         
    # Latent word embeddings
    emb_dim = 32
    glove_file = "./data/glove_doom_32d.txt"
    weights, word2emb = create_glove_embedding_init(idx2word, glove_file)
    print('tf-idf stochastic matrix (%d x %d) is generated.' % (tfidf.size(0), tfidf.size(1)))
    
    return tfidf, weights

def tfidf_loading(w_emb, dictionary):
    if os.path.isfile('./data/embed_tfidf_weights.pkl') == True:
        print("Loading embedding tfidf and weights from file")
        with open('./data/embed_tfidf_weights.pkl', 'rb') as f:
            w_emb = torch.load(f)
        print("Load embedding tfidf and weights from file successfully")
    else:
        print("Embedding tfidf and weights haven't been saving before")
        tfidf, weights = tfidf_from_instructions(dictionary)
        w_emb.init_embedding('./data/glove_doom_32d.npy', tfidf, weights)
        with open('./data/embed_tfidf_weights.pkl', 'wb') as f:
            torch.save(w_emb, f)
        print("Saving embedding with tfidf and weights successfully")
    return w_emb

class WordEmbedding(nn.Module):
    """Word Embedding
    The ntoken-th dim is used for padding_idx, which agrees *implicitly*
    with the definition in Dictionary.
    """
    def __init__(self, ntoken, emb_dim, dropout, op=''):
        super(WordEmbedding, self).__init__()
        self.op = op
        self.emb = nn.Embedding(ntoken+1, emb_dim, padding_idx=ntoken)
        if 'c' in op:
            self.emb_ = nn.Embedding(ntoken+1, emb_dim, padding_idx=ntoken)
            self.emb_.weight.requires_grad = False # fixed
        self.dropout = nn.Dropout(dropout)
        self.ntoken = ntoken
        self.emb_dim = emb_dim

    def init_embedding(self, np_file, tfidf=None, tfidf_weights=None):
        weight_init = torch.from_numpy(np.load(np_file))
        # print(self.ntoken, self.emb_dim) #17,32
        # print(weight_init.shape) #(17, 32)
        assert weight_init.shape == (self.ntoken, self.emb_dim)
        self.emb.weight.data[:self.ntoken] = weight_init
        if tfidf is not None:
            if 0 < tfidf_weights.size:
                weight_init = torch.cat([weight_init.float(), torch.from_numpy(tfidf_weights).float()], 0)
                weight_init = torch.from_numpy(tfidf_weights).float()
            
            weight_init = tfidf.matmul(weight_init) # (N x N') x (N', F)
            self.emb_.weight.requires_grad = True
        if 'c' in self.op:
            self.emb_.weight.data[:self.ntoken] = weight_init.clone()

    def forward(self, x):
        emb = self.emb(x)
        if 'c' in self.op:
            emb = torch.cat((emb, self.emb_(x)), 2)
        emb = self.dropout(emb)
        return emb

class SentenceEmbedding(nn.Module):
    def __init__(self, in_dim, num_hid, nlayers, bidirect, dropout, rnn_type='GRU'):
        """Module for question embedding
        """
        super(SentenceEmbedding, self).__init__()
        assert rnn_type == 'LSTM' or rnn_type == 'GRU'
        rnn_cls = nn.LSTM if rnn_type == 'LSTM' else nn.GRU if rnn_type == 'GRU' else None
        
        self.rnn = rnn_cls(
            in_dim, num_hid, nlayers,
            bidirectional=bidirect,
            dropout=dropout,
            batch_first=True)

        self.in_dim = in_dim
        self.num_hid = num_hid
        self.nlayers = nlayers
        self.rnn_type = rnn_type
        self.ndirections = 1 + int(bidirect)
        
    def init_hidden(self, batch):
        # just to get the type of tensor
        weight = next(self.parameters()).data
        hid_shape = (self.nlayers * self.ndirections, batch, self.num_hid // self.ndirections)
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(*hid_shape).zero_()),
                    Variable(weight.new(*hid_shape).zero_()))
        else:
            return Variable(weight.new(*hid_shape).zero_())

    def forward(self, x):
        # x: [batch, sequence, in_dim]
        batch = x.size(0)
        hidden = self.init_hidden(batch)
        output, hidden = self.rnn(x, hidden)
        
        if self.ndirections == 1:
            return output[:, -1]
    
        forward_ = output[:, -1, :self.num_hid]
        backward = output[:, 0, self.num_hid:]
        
        return torch.cat((forward_, backward), dim=1)

    def forward_all(self, x):
        # x: [batch, sequence, in_dim]
        batch = x.size(0)
        hidden = self.init_hidden(batch)
        output, hidden = self.rnn(x, hidden)
        
        return output


class SentenceSelfAttention(nn.Module):
    def __init__(self, num_hid, dropout):
        super(SentenceSelfAttention, self).__init__()
        self.num_hid = num_hid
        self.drop = nn.Dropout(dropout)
        self.W1_self_att_q = FCNet(dims=[num_hid, num_hid], dropout=dropout,
                                   act=None)
        self.W2_self_att_q = FCNet(dims=[num_hid, 1], act=None)

    def forward(self, ques_feat):
        '''
        ques_feat: [batch, 14, num_hid]
        '''
        batch_size = ques_feat.shape[0]
        q_len = ques_feat.shape[1]

        # (batch*14,num_hid)
        ques_feat_reshape = ques_feat.contiguous().view(-1, self.num_hid)
        # (batch, 14)
        atten_1 = self.W1_self_att_q(ques_feat_reshape)
        atten_1 = torch.tanh(atten_1)
        atten = self.W2_self_att_q(atten_1).view(batch_size, q_len)
        # (batch, 1, 14)
        weight = F.softmax(atten.t(), dim=1).view(-1, 1, q_len)
        ques_feat_self_att = torch.bmm(weight, ques_feat)
        ques_feat_self_att = ques_feat_self_att.view(-1, self.num_hid)
        # (batch, num_hid)
        ques_feat_self_att = self.drop(ques_feat_self_att)
        return ques_feat_self_att