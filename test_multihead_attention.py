import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn
import torchvision as TV

from einops import rearrange

class MultiHeadRelationalModule(torch.nn.Module):
    def __init__(self):
        super(MultiHeadRelationalModule, self).__init__()
        self.conv1_ch = 128 
        self.conv2_ch = 64
        self.conv3_ch = 64
        # self.conv4_ch = 30
        # self.H = 168
        # self.W = 300
        self.node_size = 64
        self.lin_hid = 100
        self.out_dim = 5
        self.ch_in = 3
        self.sp_coord_dim = 2
        # self.N = int(7**2)
        self.N = 32
        self.n_heads = 3
        
        self.conv1 = nn.Conv2d(self.ch_in, self.conv1_ch, kernel_size=8, stride=4)  # A
        self.conv2 = nn.Conv2d(self.conv1_ch, self.conv2_ch, kernel_size=4, stride=4)
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
        
        self.linear1 = nn.Linear(self.n_heads * self.node_size, self.node_size)
        self.norm1 = nn.LayerNorm([self.N,self.node_size], elementwise_affine=False)
        self.linear2 = nn.Linear(self.node_size, self.out_dim)
    
    def forward(self,x):
        N, Cin, H, W = x.shape
        x = self.conv1(x) 
        x = torch.relu(x)
        x = self.conv2(x) 
        x = torch.relu(x) 
        x = self.conv3(x)
        x = torch.relu(x) # 1, 64, 8, 17

        
        with torch.no_grad(): 
            self.conv_map = x.clone() #C
        _,_,cH,cW = x.shape
        xcoords = torch.arange(cW).repeat(cH,1).float() / cW
        ycoords = torch.arange(cH).repeat(cW,1).transpose(1,0).float() / cH
        spatial_coords = torch.stack([xcoords,ycoords],dim=0)
        spatial_coords = spatial_coords.unsqueeze(dim=0)
        spatial_coords = spatial_coords.repeat(N,1,1,1)
        x = torch.cat([x,spatial_coords],dim=1)
        x = x.permute(0,2,3,1)
        x = x.flatten(1,2) # 1, 136, 66
        
        K = rearrange(self.k_proj(x), "b n (head d) -> b head n d", head=self.n_heads)
        K = self.k_norm(K) 
        
        Q = rearrange(self.q_proj(x), "b n (head d) -> b head n d", head=self.n_heads)
        Q = self.q_norm(Q) 
        
        V = rearrange(self.v_proj(x), "b n (head d) -> b head n d", head=self.n_heads)
        V = self.v_norm(V) 
        
        A = torch.nn.functional.elu(self.q_lin(Q) + self.k_lin(K)) #D
        A = self.a_lin(A)
        print(A.shape)
        A = torch.nn.functional.softmax(A,dim=3) 
        with torch.no_grad():
            self.att_map = A.clone() #E
            
            print(self.att_map[0][2][26].shape)
            print(self.att_map.shape)
            plt.imshow(self.att_map[0][2][26].view(4,8), cmap='gray')
            plt.show()

            exit()
        E = torch.einsum('bhfc,bhcd->bhfd',A,V) #F
        E = rearrange(E, 'b head n d -> b n (head d)')
        E = self.linear1(E)
        E = torch.relu(E)
        E = self.norm1(E)
        E = E.max(dim=1)[0]
        y = self.linear2(E)
        y = torch.nn.functional.elu(y)
        
        return y

import gym
from gym_minigrid.minigrid import *
from gym_minigrid.wrappers import FullyObsWrapper, ImgObsWrapper
from skimage.transform import resize
import cv2

def prepare_state(x): #A
    x = cv2.resize(x, (300, 168))
    
    ns = torch.from_numpy(x).float().permute(2,0,1).unsqueeze(dim=0)#
    maxv = ns.flatten().max()
    ns = ns / maxv
    return ns

def get_minibatch(replay,size): #B
    batch_ids = np.random.randint(0,len(replay),size)
    batch = [replay[x] for x in batch_ids] #list of tuples
    state_batch = torch.cat([s for (s,a,r,s2,d) in batch],)
    action_batch = torch.Tensor([a for (s,a,r,s2,d) in batch]).long()
    reward_batch = torch.Tensor([r for (s,a,r,s2,d) in batch])
    state2_batch = torch.cat([s2 for (s,a,r,s2,d) in batch],dim=0)
    done_batch = torch.Tensor([d for (s,a,r,s2,d) in batch])
    return state_batch,action_batch,reward_batch,state2_batch, done_batch

def get_qtarget_ddqn(qvals,r,df,done): #C
    targets = r + (1-done) * df * qvals
    return targets


def lossfn(pred,targets,actions): #A
    loss = torch.mean(torch.pow(\
                                targets.detach() -\
                                pred.gather(dim=1,index=actions.unsqueeze(dim=1)).squeeze()\
                                ,2),dim=0)
    return loss
  
def update_replay(replay,exp,replay_size): #B
    r = exp[2]
    N = 1
    if r > 0:
        N = 50
    for i in range(N):
        replay.append(exp)
    return replay

action_map = { #C
    0:0, 
    1:1,
    2:2,
    3:3,
    4:5,
}


from collections import deque
env = ImgObsWrapper(gym.make('MiniGrid-DoorKey-5x5-v0')) #A
state = prepare_state(env.reset()) 

GWagent = MultiHeadRelationalModule() #B
pytorch_total_params = sum(p.numel() for p in GWagent.parameters())
print(pytorch_total_params)
Tnet = MultiHeadRelationalModule() #C
maxsteps = 400 #D
env.max_steps = maxsteps
env.env.max_steps = maxsteps

epochs = 200#50000
replay_size = 9000
batch_size = 50
lr = 0.0005
gamma = 0.99
replay = deque(maxlen=replay_size) #E
opt = torch.optim.Adam(params=GWagent.parameters(),lr=lr)
eps = 0.5
update_freq = 100
for i in range(epochs):
    print(i)
    pred = GWagent(state)
    action = int(torch.argmax(pred).detach().numpy())
    if np.random.rand() < eps: #F
        action = int(torch.randint(0,5,size=(1,)).squeeze())
    action_d = action_map[action]
    state2, reward, done, info = env.step(action_d)
    reward = -0.01 if reward == 0 else reward #G
    state2 = prepare_state(state2)
    exp = (state,action,reward,state2,done)
    
    replay = update_replay(replay,exp,replay_size)
    if done:
        state = prepare_state(env.reset())
    else:
        state = state2
    if len(replay) > batch_size:
        
        opt.zero_grad()
        
        state_batch,action_batch,reward_batch,state2_batch,done_batch = get_minibatch(replay,batch_size)
        
        q_pred = GWagent(state_batch).cpu()
        astar = torch.argmax(q_pred,dim=1)
        qs = Tnet(state2_batch).gather(dim=1,index=astar.unsqueeze(dim=1)).squeeze()
        
        targets = get_qtarget_ddqn(qs.detach(),reward_batch.detach(),gamma,done_batch)
        
        loss = lossfn(q_pred,targets.detach(),action_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(GWagent.parameters(), max_norm=1.0) #H
        opt.step()
    if i % update_freq == 0: #I
        Tnet.load_state_dict(GWagent.state_dict())


state_ = env.reset()
state = prepare_state(state_)
GWagent(state)
plt.imshow(env.render('rgb_array'))
plt.imshow(state[0].permute(1,2,0).detach().numpy())
plt.show()
head, node = 2, 26
print(GWagent.att_map[0][head][node].shape)
plt.imshow(GWagent.att_map[0][head][node].view(4,8))

plt.show()