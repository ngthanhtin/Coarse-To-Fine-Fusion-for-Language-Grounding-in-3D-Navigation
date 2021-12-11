# Visualizing and Understanding Atari Agents | Sam Greydanus | 2017 | MIT License

from __future__ import print_function
import warnings ; warnings.filterwarnings('ignore') # mute warnings, live dangerously ;)

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np

def rollout(model, env, max_ep_len=3e3):
    history = {'ins': [], 'instruction': [], 'logits': [], 'values': [], 'outs': [], 'hx': [], 'cx': [], 'tx': []}
    
    (obs, depth, instruction), _, _ = env.reset()
    print("Instruction for this rollout: ", instruction)
    # Getting indices of the words in the instruction
    instruction_idx = []
    for word in instruction.split(" "):
        instruction_idx.append(env.word_to_idx[word])
    instruction_idx = np.array(instruction_idx).astype(float)
    obs = torch.from_numpy(obs).float()/255.0
    instruction_idx = torch.from_numpy(instruction_idx).view(1, -1).float()

    episode_length, epr, eploss, done  = 0, 0, 0, False # bookkeeping
    cx = torch.Tensor(torch.zeros(1, 256))
    hx = torch.Tensor(torch.zeros(1, 256))

    while not done and episode_length <= max_ep_len:
        episode_length += 1
        if done:
            cx = torch.Tensor(torch.zeros(1, 256))
            hx = torch.Tensor(torch.zeros(1, 256))
        else:
            cx = torch.Tensor(cx.data)
            hx = torch.Tensor(hx.data)
        tx = torch.Tensor(torch.from_numpy(np.array([episode_length])).float()) #.long()
        instruction_idx = instruction_idx.float()

        value, logit, (hx, cx) = model( (torch.Tensor(obs.unsqueeze(0)),
                    torch.Tensor(instruction_idx), (tx, hx, cx)))
        hx, cx = torch.Tensor(hx.data), torch.Tensor(cx.data)
        prob = F.softmax(logit)

        action = prob.max(1)[1].data # prob.multinomial().data[0] # 
        (obs, depth, _), reward, done = env.step(action.numpy()[0])
        obs = torch.from_numpy(obs).float()/255.0

        epr += reward

        # save info!
        history['ins'].append(obs)
        history['instruction'].append(instruction_idx)
        history['hx'].append(hx)
        history['cx'].append(cx)
        history['tx'].append(tx)
        history['logits'].append(logit.data.numpy()[0])
        history['values'].append(value.data.numpy()[0])
        history['outs'].append(prob.data.numpy()[0])
        print('step # {}, reward {:.0f}'.format(episode_length, epr))

    return history