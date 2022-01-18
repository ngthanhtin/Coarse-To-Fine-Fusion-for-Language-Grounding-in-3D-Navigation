"""
Based on Convolution Attention as in: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8658389 to develop a new fusion method for Instruction Following task
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

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

    def forward(self, img_feats, text_feat):
        # resize convolution layers
        new_img_feats = []
        for img_feat in img_feats:
            new_img_feat = F.interpolate(img_feat, (17, 8))
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