"""
Implement Convolution Attention as in: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8658389
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# Convolved Attention
class ConvolvedAttention(nn.Module):
    def __init__(self, num_sent_fc, img_feat_size, ques_feat_size, fc_size):
        super(ConvolvedAttention, self).__init__()

        self.img_feat_size = img_feat_size
        self.ques_feat_size = ques_feat_size
        self.fc_size = fc_size

        self.layers = nn.ModuleList()
        self.num_sent_fc = num_sent_fc

        for i in range(num_sent_fc):
            self.layers.append(nn.Linear(ques_feat_size, fc_size, bias=True))

        self.attention_maps = None

    def forward(self, img_feat, text_feat):
        """
        img_feat: (B, C, W, H)
        question_feat: (B, Dim) (last hidden layer of LSTM)
        """
        for i in range(self.num_sent_fc):
            sent_fc = self.layers[i](text_feat)
            sent_fc = sent_fc.unsqueeze(2).unsqueeze(2) # B, Dim, 1, 1
            attention = F.conv2d(img_feat, sent_fc, stride=(1,1))
            
            if i == 0:
                self.attention_maps = attention
            else:
                self.attention_maps = torch.cat([self.attention_maps, attention], 0)

        return self.attention_maps