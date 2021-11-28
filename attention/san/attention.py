import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm

# Stacked Attention
class StackedAttention(nn.Module):
    def __init__(self, num_stacks, img_feat_size, ques_feat_size, att_size, output_size, drop_ratio):
        super(StackedAttention, self).__init__()

        self.img_feat_size = img_feat_size
        self.ques_feat_size = ques_feat_size
        self.att_size = att_size
        self.output_size = output_size
        self.drop_ratio = drop_ratio
        self.num_stacks = num_stacks
        self.layers = nn.ModuleList()

        self.dropout = nn.Dropout(drop_ratio)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

        self.fc11 = nn.Linear(ques_feat_size, att_size, bias=True)
        self.fc12 = nn.Linear(img_feat_size, att_size, bias=False) #bias false
        self.fc13 = nn.Linear(att_size, 1, bias=True)

        for stack in range(num_stacks - 1):
            self.layers.append(nn.Linear(att_size, att_size, bias=True))
            self.layers.append(nn.Linear(img_feat_size, att_size, bias=False)) # bias false
            self.layers.append(nn.Linear(att_size, 1, bias=True))

    def forward(self, img_feat, ques_feat, v_mask=True):
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

        weight_sum_1 = torch.sum(img_att_1, dim=1)

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
        for stack in range(self.num_stacks - 1):
            ques_embs.append(self.layers[3 * stack + 0](us[-1]))
            img_embs.append(self.layers[3 * stack + 1](img_feat))

            # Compute attention distribution
            hs.append(self.tanh(ques_embs[-1].view(B, -1, self.att_size) + img_embs[-1]))
            h_embs.append(self.layers[3*stack + 2](self.dropout(hs[-1])))
            # Mask actual bounding box sizes before calculating softmax
            if v_mask:
                mask = (0 == img_embs[-1].abs().sum(2)).unsqueeze(2).expand(h_embs[-1].size())
                h_embs[-1].data.masked_fill_(mask.data, -float('inf'))
            ps.append(self.softmax(h_embs[-1]))

            #  Compute weighted sum
            img_atts.append(img_embs[-1] * ps[-1])
            weight_sums.append(torch.sum(img_atts[-1], dim=1))

            # Combine with previous stack
            ux = us[-1] + weight_sums[-1]

            # Combine with previous stack by multiple
            us.append(ux)

        return us[-1]


# Stacked Attention 2
class StackedAttention_2(nn.Module):
    def __init__(self, num_stacks, img_feat_size, ques_feat_size, att_size, output_size, drop_ratio):
        super(StackedAttention_2, self).__init__()

        self.img_feat_size = img_feat_size
        self.ques_feat_size = ques_feat_size
        self.att_size = att_size
        self.output_size = output_size
        self.drop_ratio = drop_ratio
        self.num_stacks = num_stacks
        self.layers = nn.ModuleList()

        self.dropout = nn.Dropout(drop_ratio)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

        self.Wu = nn.Linear(ques_feat_size, att_size, bias=True)
        self.Wv = nn.Conv2d(img_feat_size, att_size, kernel_size=1, padding=0)
        self.Wp = nn.Conv2d(att_size, 1, kernel_size=1, padding=0)

        self.attention_maps = []

        for stack in range(num_stacks - 1):
            self.layers.append(nn.Linear(att_size, att_size, bias=True))
            self.layers.append(nn.Conv2d(img_feat_size, att_size, kernel_size=1, padding=0))
            self.layers.append(nn.Conv2d(att_size, 1, kernel_size=1, padding=0))

    def forward(self, img_feat, ques_feat, v_mask=True):
        """
        Input: img_feat: NxDxHxW
               ques_feat: NxD
        """

        # Stack 1
        N, K = ques_feat.size(0), self.att_size
        D, H, W = img_feat.size(1), img_feat.size(2), img_feat.size(3)
        ques_emb_1 = self.Wu(ques_feat) # Nx(att_size)
        img_emb_1 = self.Wv(img_feat) # Nx (att_size) x HxW
        # Compute attention distribution 
        h1 = self.tanh(ques_emb_1.view(N, self.att_size, 1, 1).expand(N, K, H, W) + img_emb_1)
        h1_emb = self.Wp(self.dropout(h1))
    
        # Mask actual bounding box sizes before calculating softmax
        if v_mask:
            mask = (0 == img_emb_1.abs().sum(2)).unsqueeze(2).expand(h1_emb.size())
            h1_emb.data.masked_fill_(mask.data, -float('inf'))

        p1 = self.softmax(h1_emb.view(N, H*W)).view(N, 1, H, W)
        self.attention_maps.append(p1.data.clone())

        #  Compute weighted sum
        img_tilde = (p1.expand_as(img_emb_1)*img_emb_1).sum(3).sum(2).view(N, K)

        # Combine with question vector
        u1 = img_tilde + ques_feat

        # Other stacks
        us = []
        ques_embs = []
        img_embs  = []
        hs = []
        h_embs =[]
        ps  = []
        img_tildes = []

        us.append(u1)
        for stack in range(self.num_stacks - 1):
            ques_embs.append(self.layers[3 * stack + 0](us[-1]))
            img_embs.append(self.layers[3 * stack + 1](img_feat))

            # Compute attention distribution
            hs.append(self.tanh(ques_embs[-1].view(N, self.att_size, 1, 1).expand(N, K, H, W) + img_embs[-1]))
            h_embs.append(self.layers[3*stack + 2](self.dropout(hs[-1])))
            # Mask actual bounding box sizes before calculating softmax
            if v_mask:
                mask = (0 == img_embs[-1].abs().sum(2)).unsqueeze(2).expand(h_embs[-1].size())
                h_embs[-1].data.masked_fill_(mask.data, -float('inf'))
        
            ps.append(self.softmax(h_embs[-1].view(N, H*W)).view(N, 1, H, W))
            self.attention_maps.append(ps[-1].data.clone())
            #  Compute weighted sum
            img_tildes.append((ps[-1].expand_as(img_embs[-1]) * img_embs[-1]).sum(3).sum(2).view(N, K))
            
            # Combine with previous stack
            ux = img_tildes[-1] + us[-1]

            # Combine with previous stack by multiple
            us.append(ux)
        
        return us[-1]