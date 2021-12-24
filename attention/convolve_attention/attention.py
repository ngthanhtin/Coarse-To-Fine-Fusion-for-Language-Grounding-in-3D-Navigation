"""
Implement Convolution Attention as in: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8658389
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
# Convolved Attention
class ConvolvedAttention(nn.Module):
    def __init__(self, num_sent_fc, img_feat_size, ques_feat_size, fc_size):
        super(ConvolvedAttention, self).__init__()

        self.img_feat_size = img_feat_size
        self.ques_feat_size = ques_feat_size
        self.fc_size = fc_size

        self.layers = nn.ModuleList()
        self.num_sent_fc = num_sent_fc

        # for i in range(num_sent_fc):
        #     self.layers.append(nn.Linear(ques_feat_size, fc_size, bias=True))

        self.attention_maps = None

        self.channels = [128,64,64]
        for k in range(num_sent_fc):
            for i in range(len(self.channels)):
                self.layers.append(nn.Linear(ques_feat_size, self.channels[i], bias=True))

        # self.R = nn.Parameter(torch.rand(1, fc_size, fc_size))
        # self.R_list = []
        # for i in range(num_sent_fc):
        #     self.R_list.append(nn.Parameter(torch.rand(1, fc_size, fc_size)))

    # def image_encoding_filter(self, text_features, img_features, i):
    #     """
    #     img_features: after convert num channels to self attention size
    #     text_features: after convert num dims to self attention size
    #     """
        
    #     B, D, H, W = img_features.size(0), img_features.size(1), img_features.size(2), img_features.size(3)
    #     #### text_features - (bs, D1)
    #     #### img_features - (bs, channel, H, W)
    #     img_features = img_features.view(img_features.size(0), img_features.size(1), -1)
    #     img_features_tran = img_features
        
    #     text_features = text_features.unsqueeze(0)
    #     text_features = text_features.expand(-1, H*W, -1)

    #     affinity_matrix_int = torch.bmm(text_features, self.R_list[i])
    #     affinity_matrix = torch.bmm(affinity_matrix_int, img_features_tran)
        
    #     affinity_matrix_sum = torch.sum(affinity_matrix, dim=1)
    #     affinity_matrix_sum = torch.unsqueeze(affinity_matrix_sum, dim=1)
    #     alpha_h = affinity_matrix/affinity_matrix_sum

    #     alpha_h_tran = alpha_h.permute(0,2,1)
    #     a_h = torch.bmm(alpha_h_tran, text_features)
    
    #     cos = nn.CosineSimilarity(dim=2, eps=1e-6)
    #     pdist = nn.PairwiseDistance(p=2)
    #     epsilon = 0.3
    #     # gates = (1 - cos(img_features.cpu(), a_h.cpu())).to(self.device)
    #     img_features = img_features.permute(0,2,1)
    #     # gates = epsilon * (1-pdist(img_features.squeeze().cpu(), a_h.squeeze().cpu()).unsqueeze(0)) +  (1 - epsilon) * (1-cos(img_features.cpu(), a_h.cpu()))
    #     gates = (1-cos(img_features.cpu(), a_h.cpu()))
    #     # gates = gates.to(self.device)

    #     gated_image_features = a_h * gates[:, :, None]     
    #     return gated_image_features.view(B, D, H, W)


    def forward(self, img_feats, text_feat):
        """
        img_feats: list of convolution layers [(B, C, W, H), ..]
        question_feat: (B, Dim) (last hidden layer of LSTM)
        """
        
        new_img_feats = []
        for img_feat in img_feats:
            img_feat = img_feat.permute(0,2,3,1)
            new_img_feat = cv2.resize(img_feat[0].cpu().detach().numpy(), (17, 8), interpolation=cv2.INTER_LINEAR)
            new_img_feat = torch.Tensor(new_img_feat).unsqueeze(0)
            new_img_feat = new_img_feat.permute(0,3,1,2)
            new_img_feats.append(new_img_feat)

        for k in range(self.num_sent_fc):
            for i in range(len(self.channels)):
                sent_fc = self.layers[i + k*len(self.channels)](text_feat)
                sent_fc = sent_fc.unsqueeze(2).unsqueeze(2) # B, Dim, 1, 1
                attention = F.conv2d(new_img_feats[i], sent_fc, stride=(1,1))
                
                if i + k*len(self.channels) == 0:
                    self.attention_maps = attention
                else:
                    self.attention_maps = torch.cat([self.attention_maps, attention], 0)
        
        return self.attention_maps


    # def forward(self, img_feat, text_feat):
    #     """
    #     img_feat: (B, C, W, H)
    #     question_feat: (B, Dim) (last hidden layer of LSTM)
    #     """
    #     for i in range(self.num_sent_fc):
    #         sent_fc = self.layers[i](text_feat)
    #         sent_fc = sent_fc.unsqueeze(2).unsqueeze(2) # B, Dim, 1, 1
    #         attention = F.conv2d(img_feat, sent_fc, stride=(1,1))
            
    #         if i == 0:
    #             self.attention_maps = attention
    #         else:
    #             self.attention_maps = torch.cat([self.attention_maps, attention], 0)

    #     return self.attention_maps