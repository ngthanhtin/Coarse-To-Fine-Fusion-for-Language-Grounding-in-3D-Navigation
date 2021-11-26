import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from language_model import tfidf_loading, WordEmbedding, SentenceEmbedding, SentenceSelfAttention
from attention import StackedAttention, BiAttention
from bc import BCNet
from fc import FCNet
import cv2
from einops import rearrange

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
        # self.convert = nn.Linear(55488, 8704)
        #init auto encoder
        if args.auto_encoder:
            self.ae_model = ae_model
        #language model
        self.w_emb = WordEmbedding(args.vocab_size, 32, 0.0) # , op='c'
        # self.w_emb = tfidf_loading(self.w_emb, args.dictionary)
        self.s_emb = SentenceEmbedding(32, 256, 1, False, 0.0, 'GRU')
        # self.s_emb_att = SentenceSelfAttention(256, 0)

        #attention
        if args.attention == 'san':
            self.v_att = StackedAttention(1, 136, 256 , 256, 2, 0.0) 
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
        # multihead attention
        self.conv1_ch = 128 
        self.conv2_ch = 64
        self.conv3_ch = 64
        
        self.node_size = 64
        self.lin_hid = 100
        self.out_dim = 5
        self.ch_in = 3
        self.sp_coord_dim = 2
        self.N = int(7**2)
        self.N = 136 # 32
        self.n_heads = 3
        
        self.conv1 = nn.Conv2d(self.ch_in, self.conv1_ch, kernel_size=8, stride=4)  # A
        self.conv2 = nn.Conv2d(self.conv1_ch, self.conv2_ch, kernel_size=4, stride=2) # 4
        self.conv3 = nn.Conv2d(self.conv2_ch, self.conv3_ch, kernel_size=4, stride=2)

        self.proj_shape = (self.conv2_ch+self.sp_coord_dim,self.n_heads * self.node_size)
        self.k_proj = nn.Linear(*self.proj_shape)
        self.q_proj = nn.Linear(*self.proj_shape)
        self.v_proj = nn.Linear(*self.proj_shape)

        self.k_lin = nn.Linear(self.node_size,self.N) #B
        self.q_lin = nn.Linear(self.node_size,self.N)
        self.a_lin = nn.Linear(self.N,self.N)
        
        self.node_shape = (self.n_heads, self.N,self.node_size)
        self.k_norm = nn.LayerNorm(self.node_shape, elementwise_affine=True)
        self.q_norm = nn.LayerNorm(self.node_shape, elementwise_affine=True)
        self.v_norm = nn.LayerNorm(self.node_shape, elementwise_affine=True)

        self.linear1 = nn.Linear(self.n_heads * self.node_size, 256) # self.node_size
        self.norm1 = nn.LayerNorm([self.N,self.node_size * 4], elementwise_affine=False)

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
            x1 = encoder.clone()

            # multihead attention
            with torch.no_grad(): 
                self.conv_map = x1.clone() #C
            _,_,cH,cW = x1.shape
            xcoords = torch.arange(cW).repeat(cH,1).float() / cW
            ycoords = torch.arange(cH).repeat(cW,1).transpose(1,0).float() / cH
            spatial_coords = torch.stack([xcoords,ycoords],dim=0)
            spatial_coords = spatial_coords.unsqueeze(dim=0)
            spatial_coords = spatial_coords.repeat(N,1,1,1)
            x1 = torch.cat([x1,spatial_coords],dim=1)
            x1 = x1.permute(0,2,3,1)
            x1 = x1.flatten(1,2)
            
            K = rearrange(self.k_proj(x1), "b n (head d) -> b head n d", head=self.n_heads)
            K = self.k_norm(K) 
            
            Q = rearrange(self.q_proj(x1), "b n (head d) -> b head n d", head=self.n_heads)
            Q = self.q_norm(Q) 
            
            V = rearrange(self.v_proj(x1), "b n (head d) -> b head n d", head=self.n_heads)
            V = self.v_norm(V) 
            A = torch.nn.functional.elu(self.q_lin(Q) + self.k_lin(K)) #D # elu
            A = self.a_lin(A)
            A = torch.nn.functional.softmax(A,dim=3) 
            with torch.no_grad():
                self.att_map = A.clone() #E
                # print(self.att_map.shape)
            
            E = torch.einsum('bhfc,bhcd->bhfd',A,V) #F
            E = rearrange(E, 'b head n d -> b n (head d)')
            E = self.linear1(E)
            E = self.prelu(E)
            E = self.norm1(E)
            E = E.max(dim=1)[0]

            #get the auto encoder representation  
            ae_v_emb = encoder.view(encoder.shape[0], -1) 
            x_emb1 = E.view(1, -1)

        else:
            x, input_inst, (tx, hx, cx) = inputs
            # Get the image representation
            N, Cin, H, W = x.shape
            x = self.conv1(x) 
            x = self.prelu(x)
            x = self.conv2(x) 
            x = self.prelu(x) 
            x = self.conv3(x)
            x = self.prelu(x)
            # x1 = x.clone()
            
            # multihead attention
            with torch.no_grad(): 
                self.conv_map = x1.clone() #C
            _,_,cH,cW = x1.shape

            xcoords = torch.arange(cW).repeat(cH,1).float() / cW
            ycoords = torch.arange(cH).repeat(cW,1).transpose(1,0).float() / cH
            spatial_coords = torch.stack([xcoords,ycoords],dim=0)
            spatial_coords = spatial_coords.unsqueeze(dim=0)
            spatial_coords = spatial_coords.repeat(N,1,1,1)
            x1 = torch.cat([x1,spatial_coords],dim=1)
            x1 = x1.permute(0,2,3,1)
            x1 = x1.flatten(1,2)
            
            K = rearrange(self.k_proj(x1), "b n (head d) -> b head n d", head=self.n_heads)
            K = self.k_norm(K) 
            
            Q = rearrange(self.q_proj(x1), "b n (head d) -> b head n d", head=self.n_heads)
            Q = self.q_norm(Q) 
            
            V = rearrange(self.v_proj(x1), "b n (head d) -> b head n d", head=self.n_heads)
            V = self.v_norm(V) 
            
            A = torch.nn.functional.elu(self.q_lin(Q) + self.k_lin(K)) #D # elu
            A = self.a_lin(A)
            
            A = torch.nn.functional.softmax(A,dim=3) 
            with torch.no_grad():
                self.att_map = A.clone() #E
            
            E = torch.einsum('bhfc,bhcd->bhfd',A,V) #F
            E = rearrange(E, 'b head n d -> b n (head d)')
            E = self.linear1(E)
            E = self.prelu(E)
            E = self.norm1(E)
            E = E.max(dim=1)[0]

            x_emb = E.view(1, -1)
            
            # x_emb = x

        tx = tx.long()
        
        if self.args.auto_encoder:
            x_emb = ae_v_emb
            # x_emb = torch.cat((x_emb, ae_v_emb), 1)
        
        w_emb = self.w_emb(input_inst.long())
        if self.args.attention == "san":
            s_emb = self.s_emb(w_emb)
        elif self.args.attention == "ban":
            s_emb = self.s_emb.forward_all(w_emb)
            x_emb = x_emb.view(1, 34, s_emb.size(2))
        
        # s_emb = s_emb.unsqueeze(dim=0)   
        # s_emb = self.s_emb_att(s_emb)
        

        if self.args.attention == "san":
            # x_emb = self.convert(x_emb)
            # att = self.v_att(x_emb, s_emb, v_mask=False)
            # att = att * x_emb1
            # att = torch.cat((att, x_emb1), 1)

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

    