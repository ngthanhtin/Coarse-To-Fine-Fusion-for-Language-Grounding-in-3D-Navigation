import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm

# Dual Attention
class DualAttention(nn.Module):
    def __init__(self, img_feat_size, ques_feat_size, att_size, output_size, drop_ratio):
        super(DualAttention, self).__init__()

        self.img_feat_size = img_feat_size
        self.ques_feat_size = ques_feat_size
        self.att_size = att_size
        self.output_size = output_size
        self.drop_ratio = drop_ratio
        self.layers = nn.ModuleList()

        self.dropout = nn.Dropout(drop_ratio)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

        self.fc11 = nn.Linear(ques_feat_size, att_size, bias=True)
        self.fc12 = nn.Linear(img_feat_size, att_size, bias=False) #bias false
        self.fc13 = nn.Linear(att_size, 1, bias=True)

    def forward(self, img_feat, ques_feat, v_mask=True):
        # import matplotlib.pyplot as plt
        # import numpy as np
        # import cv2
        # test_att1 = img_feat.view((64, 8, 17)) # 64,8,17
        # # test_att1 = test_att1.sum(0).unsqueeze(0)
        # test_att1 = test_att1[0].unsqueeze(0)
        # print(test_att1.shape)
        # att_image = test_att1.permute(1, 2, 0).detach().numpy()
        # att_image = cv2.resize(att_image, (300, 168)) 
        # cv2.imshow("ads", att_image)
        # cv2.waitKey(0)

        # exit()
        # Batch size
        B = ques_feat.size(0)

        # Stack 1
        ques_emb_1 = self.fc11(ques_feat)
        img_emb_1 = self.fc12(img_feat)
        # Compute attention distribution
        h1 = self.tanh(ques_emb_1.view(B, 1, self.att_size) + img_emb_1)
        h1_emb = self.fc13(self.dropout(h1))
        # Mask actual bounding box sizes before calculating softmax
        if v_mask:
            mask = (0 == img_emb_1.abs().sum(2)).unsqueeze(2).expand(h1_emb.size())
            h1_emb.data.masked_fill_(mask.data, -float('inf'))

        p1 = self.softmax(h1_emb)
        
        #  Compute weighted sum
        img_att_1 = img_emb_1*p1
        # import matplotlib.pyplot as plt
        # import numpy as np
        # import cv2
        
        # test_att1 = img_att_1.view((1, 16, 16))
        # # test_att1 = test_att1.sum(0).unsqueeze(0)
        # att_image = test_att1.permute(1, 2, 0).detach().numpy()
        # att_image = cv2.resize(att_image, (300, 168)) 
        # cv2.imshow("ads", att_image)
        # cv2.waitKey(0)

        weight_sum_1 = torch.sum(img_att_1, dim=1)

        # import matplotlib.pyplot as plt
        # import numpy as np
        # import cv2
        
        # test_att1 = weight_sum_1.view((1, 16, 16))
        # # test_att1 = test_att1.sum(0).unsqueeze(0)
        # att_image = test_att1.permute(1, 2, 0).detach().numpy()
        # att_image = cv2.resize(att_image, (300, 168)) 
        # cv2.imshow("ads", att_image)
        # cv2.waitKey(0)

        # Combine with question vector
        u1 = ques_emb_1 + weight_sum_1

        # Other stacks
        us = []
        ques_embs = []
        img_embs  = []
        hs = []
        h_embs =[]
        ps  = []
        img_atts = []
        weight_sums = []

        us.append(u1)

        return us[-1]