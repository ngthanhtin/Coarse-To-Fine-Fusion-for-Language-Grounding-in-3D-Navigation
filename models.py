import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from language_model import tfidf_loading, WordEmbedding, SentenceEmbedding
from attention import StackedAttention, BiAttention
from bc import BCNet
from fc import FCNet
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
        self.prelu = nn.PReLU() 
        #init auto encoder
        if args.auto_encoder:
            self.ae_model = ae_model
            self.convert = nn.Linear(50400, 8704)
        #language model
        self.w_emb = WordEmbedding(args.vocab_size, 32, 0.0) # , op='c'
        # self.w_emb = tfidf_loading(self.w_emb, args.dictionary)
        self.s_emb = SentenceEmbedding(32, 256, 1, False, 0.0, 'GRU')
        
        #attention
        if args.attention == 'san':
            self.v_att = StackedAttention(2, 64*8*17, 256 , 256, 2, 0.0) #dropout=0.5
        elif args.attention == 'ban':
            self.v_att = BiAttention(256, 256, 256, args.glimpse)
            # init BAN residual network
            self.b_net = []
            self.q_prj = []
            for i in range(args.glimpse): #glimpse
                self.b_net.append(BCNet(256, 256, 256, None, k=1))
                self.q_prj.append(FCNet([256, 256], '', .2))
            self.b_net = nn.ModuleList(self.b_net)
            self.q_prj = nn.ModuleList(self.q_prj)

        # Image Processing
        # self.conv1 = nn.Conv2d(3, 128, kernel_size=8, stride=4) 
        # self.conv2 = nn.Conv2d(128, 64, kernel_size=4, stride=2)
        # self.conv3 = nn.Conv2d(64, 64, kernel_size=4, stride=2)

        # Time embedding layer, helps in stabilizing value prediction
        self.time_emb_dim = 32
        self.time_emb_layer = nn.Embedding(
                args.max_episode_length+1,
                self.time_emb_dim)

        # A3C-LSTM layers
        self.linear = nn.Linear(256, 256)

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
            encoder = self.ae_model.forward_pass(ae_input)
            decoder = self.ae_model.reconstruct_pass(encoder)
            #get the auto encoder representation  
            ae_v_emb = encoder.view(encoder.shape[0], -1) 
        else:
            x, input_inst, (tx, hx, cx) = inputs
            # Get the image representation
            x = self.prelu(self.conv1(x))
            # x = F.relu(self.conv1(x))
            x =self.prelu(self.conv2(x))
            # x = F.relu(self.conv2(x))
            x_image_rep = self.prelu(self.conv3(x))
            # x_image_rep = F.relu(self.conv3(x))
            x_emb = x_image_rep.view(1, -1)
            
        tx = tx.long()
        
        if self.args.auto_encoder:
            x_emb = ae_v_emb
            # x_emb = torch.cat((x_emb, ae_v_emb), 1)
        w_emb = self.w_emb(input_inst.long())
        # with open("word.txt", "a+") as f:
        #     f.write("{}\n".format(input_inst.cpu().detach().clone().numpy()[0]))
        # with open("w_emb.txt", "a+") as f:
        #     f.write("{}\n".format(w_emb.cpu().detach().clone().numpy()[0]))

        if self.args.attention == "san":
            s_emb = self.s_emb(w_emb)
        elif self.args.attention == "ban":
            s_emb = self.s_emb.forward_all(w_emb)
            x_emb = x_emb.view(1, 34, s_emb.size(2))

        # with open("s_w_emb.txt", "a+") as f:
        #     f.write("{}\n".format(s_emb.cpu().detach().clone().numpy()[0]))  

        # exit()
        if self.args.attention == "san":
            att = self.v_att(x_emb, s_emb, v_mask=False)
        elif self.args.attention == "ban":
            b_emb = [0] * self.args.glimpse
            att, logits = self.v_att.forward_all(x_emb, s_emb) # b x g x v x q
            for g in range(self.args.glimpse):
                b_emb[g] = self.b_net[g].forward_with_weights(x_emb, s_emb, att[:,g,:,:]) # b x l x h
                atten, _ = logits[:,g,:,:].max(2)
                s_emb = self.q_prj[g](b_emb[g].unsqueeze(1)) + s_emb
            att = s_emb.sum(1)
        
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
            return self.critic_linear(x), self.actor_linear(x), (hx, cx), decoder
        return self.critic_linear(x), self.actor_linear(x), (hx, cx)

    