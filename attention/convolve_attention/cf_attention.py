"""
Based on Convolution Attention as in: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8658389 to develop a new fusion method for Instruction Following task
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

def filter_predicate(input_sent):
    """
    input_sent: the input sentence as a vector
    return: indexes of predicates in the sentence
    """
    dictionary = {'Go': 0, 'to': 1, 'the': 2, 'short': 3, 'red': 4, 'object': 5, 'torch': 6, 'green': 7, \
        'tall': 8, 'smallest': 9, 'pillar': 10, 'blue': 11, 'yellow': 12, 'skullkey': 13, 'largest': 14, 'keycard': 15, 'armor': 16}
    # colors = [4,7,11,12]
    # sizes = [3, 8]
    # objects = [5,6,10,13,15,16]
    # superelatives = [9,14]

    predicate_list = []
    for c, i in enumerate(input_sent[0]):
        # if c in colors or c in sizes or c in objects or c in superelatives:
        if i > 2 and i < 17:
            predicate_list.append(c)
    return predicate_list

# Corse to Fine Convolved Attention
class CF_ConvolvedAttention(nn.Module):
    """
    Corse to Fine Convolve Attention
    """
    def __init__(self, num_sent_fc, img_feat_size, ques_feat_size, predicate_size, fc_size):
        super(CF_ConvolvedAttention, self).__init__()

        self.img_feat_size = img_feat_size
        self.ques_feat_size = ques_feat_size
        self.fc_size = fc_size

        self.layers = nn.ModuleList()
        self.num_sent_fc = num_sent_fc

        self.attention_maps = None

        self.channels = [128,64,64]
        for k in range(num_sent_fc):
            for i in range(len(self.channels)):
                self.layers.append(nn.Linear(ques_feat_size, self.channels[i], bias=True))

    #     # use for filtering process
    #     self.softmax = nn.Softmax(dim=-1)
    #     self.psi_dim = self.ques_feat_size
    #     # self.linear_sentence = nn.Linear(ques_feat_size, self.psi_dim)
    #     self.linear_predicate = nn.Linear(predicate_size, self.psi_dim)

    # def sentence_filtering(self, predicate_feat, sentence_feat):
    #     """
    #     predicate feat: num predicate x dim of each predicate
    #     sentence feat: B x dim of a sentence
    #     return filtered sentence: B x dim psi
    #     """
    #     predicate_feat = self.linear_predicate(predicate_feat) #np, psi_dim
    #     # sentence_feat = self.linear_sentence(sentence_feat)#B, psi_dim 

    #     psi = self.softmax(torch.matmul(sentence_feat, predicate_feat.T).sum(1))
    #     channel_scaled_vector = torch.ones(self.psi_dim, 1)
    #     filter_sent_feat = torch.matmul(psi, channel_scaled_vector.T) * sentence_feat + sentence_feat # B, psi_dim

    #     return filter_sent_feat

    def forward(self, img_feats, text_feat):
        #  , word_embedding, input_instr
        """
        img_feats: list of convolution layers [(B, C, W, H), ..]
        text_feat: (B, Dim) (last hidden layer of LSTM)
        word_embedding: word embedding of the input instruction
        input_instr: input instruction as a vector
        predicate_feat: (num of predicates, dim of each predicate)
        """
        # predicate_list = filter_predicate(input_instr)
        # predicate_list_tensor = torch.Tensor(predicate_list).int()
        # # channel_scaled_vector = torch.zeros(self.psi_dim, 1)
        # predicate_feat = torch.index_select(word_embedding, 1, predicate_list_tensor)
        # predicate_feat = predicate_feat[0]
        # # filtering sentence_feat
        # text_feat = self.sentence_filtering(predicate_feat, text_feat)

        # resize convolution layers
        new_img_feats = []
        for img_feat in img_feats:
            img_feat = img_feat.permute(0,2,3,1)
            new_img_feat = cv2.resize(img_feat[0].cpu().detach().numpy(), (17, 8), interpolation=cv2.INTER_LINEAR)
            new_img_feat = torch.Tensor(new_img_feat).unsqueeze(0)
            new_img_feat = new_img_feat.permute(0,3,1,2)
            new_img_feats.append(new_img_feat)

        for k in range(self.num_sent_fc):
            attention = torch.zeros(new_img_feats[0].size(0), 1, new_img_feats[0].size(2), new_img_feats[0].size(3))
            for i in range(len(self.channels)):
                sent_fc = self.layers[i + k*len(self.channels)](text_feat)
                sent_fc = sent_fc.unsqueeze(2).unsqueeze(2) # B, Dim, 1, 1
                attention += F.conv2d(new_img_feats[i], sent_fc, stride=(1,1))
                
            if k == 0:
                self.attention_maps = attention
            else:
                self.attention_maps = torch.cat([self.attention_maps, attention], 0)
        
        return self.attention_maps