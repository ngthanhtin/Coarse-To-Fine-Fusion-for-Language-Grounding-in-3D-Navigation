import torch
from torch import nn
import torch.nn.functional as F
import torch.fft
from einops import rearrange
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import matplotlib.cm as cm
import time

def get_attention_map(img, conv_mat, is_fourier, get_mask=False):
    att_mat = conv_mat.squeeze()
    
    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(1))
    aug_att_mat = att_mat + residual_att
    # aug_att_mat = att_mat
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

    v = joint_attentions[0]
    # if is_fourier:
    #     v = joint_attentions[0]
    # else:
    #     v = joint_attentions[-1]
    print(v.shape)
    mask = v[0].reshape(4, 3).detach().numpy()
    
    if get_mask:
        result = cv2.resize(mask / mask.max(), img.size)
    else:    
        mask = cv2.resize(mask / mask.max(), img.size)[..., np.newaxis]
        # mask = cv2.resize(mask, img.size)[..., np.newaxis]
        # mask = mask.astype("int8")
        
        result = (mask * img).astype("int8")
    
    return result

def plot_attention_map(original_img, att_map, att_name):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))
    ax1.set_title('Original')
    ax2.set_title(att_name)
    _ = ax1.imshow(original_img)
    _ = ax2.imshow(att_map)
    plt.show()

class FeedForward(nn.Module):
    def __init__(self, dim, mlp_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
      
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FNetBlock(nn.Module):
  def __init__(self):
    super().__init__()
    
  def forward(self, x):
    x = torch.fft.fft(torch.fft.fft(x, dim=-1), dim=-2).real
    return x


class FNetBlockText(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x):
    x = torch.fft.fft(x, dim=-1).real
    return x

class FNetBlockImage(nn.Module):
  def __init__(self):
    super().__init__()
    
  def forward(self, x):
    x = torch.fft.fft(torch.fft.fft(x, dim=-1), dim=-2).real
    return x

class FNet(nn.Module):
    def __init__(self, dim_x, dim_y, hidden_dim, depth, mlp_dim, dropout = 0.):
        """
        dim_x: dim of image rep
        hidden_dim: channel of image rep
        dim_y: dim of words rep
        depth: depth of fnet
        mlp_dim: depth of mlps
        """
        super().__init__()
        self.layers = nn.ModuleList([])
        
        for _ in range(1):
            self.layers.append(nn.ModuleList([
                PreNorm((12, 12), FNetBlockImage()),
                PreNorm(256,FNetBlockText()),
                # PreNorm(12*12, FeedForward(12*12, 9, dropout = dropout))
                # PreNorm(dim_x * dim_x, FeedForward(dim_x * dim_x, 3, dropout = dropout))
            ]))

        self.ff = PreNorm(12*12, FeedForward(12*12, 9, dropout = dropout))
        # Gated-Attention layers
        self.attn_linear = nn.Linear(dim_y, hidden_dim)
        # pr
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.hidden_dim = hidden_dim
        self.dropout = dropout

    def forward(self, x, y):
        """
        x: image vector
        y: sentence vetor
        """
        for attn1, attn2 in self.layers:
            x = attn1(x) + x
            y = attn2(y) + y

        y_attention = torch.sigmoid(self.attn_linear(y))
        y_attention = y_attention.unsqueeze(2).unsqueeze(3)
        y_attention = y_attention.expand(1, self.hidden_dim, self.dim_x, self.dim_x)
        
        start = time.time()
        z = x*y_attention
        # print(time.time() - start)
        z = z.contiguous().view(z.size(0), 64, 12*12)

        z = self.ff(z) + z
            
        z = z.view(z.size(0), -1)

        return z

class FNet2(nn.Module):
    def __init__(self, dim_x, dim_y, hidden_dim, depth, mlp_dim, dropout = 0.):
        """
        dim_x: dim of image rep
        hidden_dim: channel of image rep
        dim_y: dim of words rep
        depth: depth of fnet
        mlp_dim: depth of mlps
        """
        super().__init__()
        self.layers = nn.ModuleList([])
        
        for _ in range(1):
            self.layers.append(nn.ModuleList([
                FNetBlockImage(),
                FNetBlockText(),
                # PreNorm(12*12, FeedForward(12*12, 9, dropout = dropout))
                # PreNorm(dim_x * dim_x, FeedForward(dim_x * dim_x, 3, dropout = dropout))
            ]))

        # self.ff = PreNorm(12*12, FeedForward(12*12, 9, dropout = dropout))
        # Gated-Attention layers
        self.attn_linear = nn.Linear(dim_y, hidden_dim)

        self.dim_x = dim_x
        self.dim_y = dim_y
        self.hidden_dim = hidden_dim
        self.dropout = dropout

    def forward(self, x, y):
        """
        x: image vector
        y: sentence vetor
        """
        for attn1, attn2 in self.layers:
            x = attn1(x)
            y = attn2(y)

        y_attention = torch.sigmoid(self.attn_linear(y))
        y_attention = y_attention.unsqueeze(2).unsqueeze(3)
        y_attention = y_attention.expand(1, self.hidden_dim, self.dim_x, self.dim_x)
        
        start = time.time()
        z = x*y_attention
        # print(time.time() - start)
        # z = z.contiguous().view(z.size(0), 64, 12*12)    
        z = z.view(z.size(0), -1)

        return z

class GatedAttention(nn.Module):
    def __init__(self, dim_x, dim_y, hidden_dim):
        super().__init__()
        """
        dim_x: width (or height) dim of conv layer
        dim_y: sentence embedding dims
        hidden_dim: linear embedding
        Ex: (12, 256, 64) # (dim_x, dim_y, hidden_dim)
        """
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.hidden_dim = hidden_dim
        # Gated-Attention layers
        self.attn_linear = nn.Linear(dim_y, hidden_dim)
        
    def forward(self, x, y):
        """
        x: conv maps
        y: sentence embedding
        """
        y_attention = torch.sigmoid(self.attn_linear(y))
        y_attention = y_attention.unsqueeze(2).unsqueeze(3)
        y_attention = y_attention.expand(1, self.hidden_dim, self.dim_x, self.dim_x)

        z = x*y_attention
        z = z.view(z.size(0), -1)

        return z

# x = torch.randn(1, 64, 12, 12)
# y = torch.randn(1, 256)
# m = FNet2(x.size()[1:], y.size(1), 64, 2, 0)
# out = m(x, y)
# print('-----')
# print(out.shape)

# exit()


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from language_model import tfidf_loading, WordEmbedding, SentenceEmbedding, SentenceSelfAttention
import cv2


def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True).expand_as(out))
    return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        # if m.bias is not None:
        #     m.bias.data.fill_(0)


class A3C_LSTM_GA(torch.nn.Module):

    def __init__(self, args, ae_model=None):
        super(A3C_LSTM_GA, self).__init__()
        self.args = args
        if args.attention == 'fga':
            self.prelu = nn.PReLU() 
        elif args.attention == 'ga':
            self.prelu = nn.ReLU() 
        else:
            print("Wrong Attention type....")

        # self.convert = nn.Linear(55488, 8704)
        #init auto encoder
        if args.auto_encoder:
            self.ae_model = ae_model
        #language model
        self.w_emb = WordEmbedding(args.vocab_size, 32, 0.0) # , op='c'
        # self.w_emb = tfidf_loading(self.w_emb, args.dictionary)
        self.s_emb = SentenceEmbedding(32, 256, 1, False, 0.0, 'GRU')
        # self.s_emb_att = SentenceSelfAttention(256, 0)

        #-------------------ATTENTION------------------#
        if args.attention == 'fga':
        # fnet attention
            self.attention = FNet(12, 256, 64, 3, 64, 0)
            
        else:
        # gated attention
            self.attention = GatedAttention(12, 256, 64)
            
        #-----------------------------------------------#

        # Image Processing
        self.conv1 = nn.Conv2d(3, 128, kernel_size=8, stride=4)  
        self.conv2 = nn.Conv2d(128, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=4, stride=2)

        # Time embedding layer, helps in stabilizing value prediction
        self.time_emb_dim = 32
        self.time_emb_layer = nn.Embedding(
                args.max_episode_length+1,
                self.time_emb_dim)

        # A3C-LSTM layers
        self.linear = nn.Linear(64*12*12, 256)

        self.lstm = nn.LSTMCell(256, 256)
        self.critic_linear = nn.Linear(256 + self.time_emb_dim, 1)
        self.actor_linear = nn.Linear(256 + self.time_emb_dim, 3)

        # Initializing weights
        # self.apply(weights_init)
        self.actor_linear.weight.data = normalized_columns_initializer(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.train()


    def forward(self, inputs):
        
        if self.args.auto_encoder:
            ae_input, x, input_inst, (tx, hx, cx) = inputs
            ae_input = F.interpolate(ae_input, (224, 224))
            encoder = self.ae_model.forward_pass(ae_input)
            decoder = self.ae_model.reconstruct_pass(encoder)
            #get the auto encoder representation  
            ae_v_emb = encoder.clone()
        else:
            x, input_inst, (tx, hx, cx) = inputs

            # Get the image representation
            N, Cin, H, W = x.shape
            x = F.interpolate(x, (224, 224))

            x = self.conv1(x) 
            x = self.prelu(x)
            x = self.conv2(x) 
            x = self.prelu(x) 
            x = self.conv3(x)
            x = self.prelu(x)
            x_emb = x

        tx = tx.long()
        
        if self.args.auto_encoder:
            x_emb = ae_v_emb
        
        w_emb = self.w_emb(input_inst.long())
        s_emb = self.s_emb(w_emb)

        if self.args.attention == 'fga':
            #f-net attention
            att = self.attention(x_emb, s_emb)            
        else:
            # gated attention
            att = self.attention(x_emb, s_emb)
    
        x = att

        # with open("abc.txt", "a+") as f:
        #     f.write("{}\n".format(x.cpu().detach().clone().numpy()[0]))
        # A3C-LSTM
        
        x = self.prelu(self.linear(x))
        # x = F.relu(self.linear(x))
        hx, cx = self.lstm(x, (hx, cx))
        time_emb = self.time_emb_layer(tx)
        x = torch.cat((hx, time_emb.view(-1, self.time_emb_dim)), 1)
        
        if self.args.auto_encoder:
            return self.critic_linear(x), self.actor_linear(x), (hx, cx), decoder, ae_input
        return self.critic_linear(x), self.actor_linear(x), (hx, cx)

    