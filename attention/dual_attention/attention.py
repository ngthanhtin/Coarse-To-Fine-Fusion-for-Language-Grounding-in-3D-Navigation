"""
Implement Dual Attention as in: https://arxiv.org/pdf/1902.01385.pdf
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# Dual Attention
class DualAttention(nn.Module):
    def __init__(self, word_to_idx, img_feat_size, ques_feat_size):
        super(DualAttention, self).__init__()

        self.word_to_idx = word_to_idx
        self.img_feat_size = img_feat_size
        self.ques_feat_size = ques_feat_size
        
        self.attn_linear_1 = nn.Linear(len(self.word_to_idx), 64) # 64 is the channel of visual features
        self.attn_linear_2 = nn.Linear(512, 64) # used for LSTM hidden features
        
        self.softmax = nn.Softmax(dim = 1)

    def make_bow_vector(self, sentence):
        vocab_size= len(self.word_to_idx)
        vec = torch.zeros(vocab_size)
        for word_idx in sentence:
            if word_idx < 0 or word_idx >= vocab_size:
                raise ValueError('out of index')
            else:
                vec[word_idx]+=1
        return vec.view(1, -1)

    def forward(self, img_feat, text_feat, input_instr):
        """
        img_feat: (B, C, W, H)
        text_feat: (B, Dim) (last hidden layer of LSTM)
        word_feat: (B, length) (length of a sentence)

        return: attention features map: (B, C, W, H)
        """
        # gated attention with BOW
        bow_vector = self.make_bow_vector(input_instr)
        bow_vector_linear = nn.ReLU(self.attn_linear_1(bow_vector))
        bow_vector_linear = bow_vector_linear.unsqueeze(2).unsqueeze(3)
        bow_vector_linear = bow_vector_linear.expand(1, 64, 8, 17)
        x_ga_1 = img_feat*bow_vector_linear
        x_ga_1 = torch.sum(x_ga_1, 1) # sum over channels
        x_spat = self.softmax(x_ga_1)

        # gated attention with LSTM hidden features
        x_sa = x_spat * img_feat
        text_feat_linear = nn.ReLU(self.attn_linear_2(text_feat))
        text_feat_linear = text_feat_linear.unsqueeze(2).unsqueeze(3)
        text_feat_linear = text_feat_linear.expand(1, 64, 8, 17)
        x_ga_2 = x_sa*text_feat_linear
        x_ga_2 = x_ga_2.view(x_ga_2.size(0), -1)

        return x_ga_2


        
