import sys
sys.path.insert(0, './')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import os, cv2, io
import pandas as pd
import numpy as np
from PIL import Image
import argparse

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from models.models import A3C_LSTM_GA
from env.env import GroundingEnv

from analysis.pertubation_saliency.rollout import rollout
from analysis.pertubation_saliency.saliency import score_frame, saliency_on_vizdoom_frame, occlude, searchlight

parser = argparse.ArgumentParser(description='Instruction Navigation in Vizdoom')

# Environment arguments
parser.add_argument('-env', '--env-type', type=str, default="single",
                    help="""Type of the environment,
                    "single", "twogoal" or "threegoal" (default: single)""")
parser.add_argument('-l', '--max-episode-length', type=int, default=30,
                    help='maximum length of an episode (default: 30)')
parser.add_argument('-d', '--difficulty', type=str, default="easy",
                    help="""Difficulty of the environment,
                    "easy", "medium" or "hard" (default: hard)""")
parser.add_argument('--living-reward', type=float, default=0,
                    help="""Default reward at each time step (default: 0,
                    change to -0.005 to encourage shorter paths)""")
parser.add_argument('--frame-width', type=int, default=300, 
                    help='Frame width (default: 300)')
parser.add_argument('--frame-height', type=int, default=168,
                    help='Frame height (default: 168)')
parser.add_argument('-v', '--visualize', type=int, default=0,
                    help="""Visualize the envrionment (default: 0,
                    use 0 for faster training)""")
parser.add_argument('--sleep', type=float, default=0,
                    help="""Sleep between frames for better
                    visualization (default: 0)""")
parser.add_argument('--scenario-path', type=str, default="maps/room.wad",
                    help="""Doom scenario file to load
                    (default: maps/room.wad)""")
parser.add_argument('--interactive', type=int, default=0,
                    help="""Interactive mode enables human to play
                    (default: 0)""")
parser.add_argument('--all-instr-file', type=str,
                    default="data/instructions_all.json",
                    help="""All instructions file
                    (default: data/instructions_all.json)""")
parser.add_argument('--train-instr-file', type=str,
                    default="data/instructions_train.json",
                    help="""Train instructions file
                    (default: data/instructions_train.json)""")
parser.add_argument('--test-instr-file', type=str,
                    default="data/instructions_test.json",
                    help="""Test instructions file
                    (default: data/instructions_test.json)""")
parser.add_argument('--object-size-file', type=str,
                    default="data/object_sizes.txt",
                    help='Object size file (default: data/object_sizes.txt)')

# A3C arguments
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--tau', type=float, default=1.00, metavar='T',
                    help='parameter for GAE (default: 1.00)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('-n', '--num-processes', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 4)')
parser.add_argument('--num-steps', type=int, default=20, metavar='NS',
                    help='number of forward steps in A3C (default: 20)')
parser.add_argument('--load', type=str, default="0",
                    help='model path to load, 0 to not reload (default: 0)')
parser.add_argument('-e', '--evaluate', type=int, default=0,
                    help="""0:Train, 1:Evaluate MultiTask Generalization
                    2:Evaluate Zero-shot Generalization (default: 0)""")
parser.add_argument('--dump-location', type=str, default="./saved/",
                    help='path to dump models and log (default: ./saved/)')

# Attention arguments
parser.add_argument('-att', '--attention', type=str, default="san",
                    help="""Type of attention,
                    "san" or "convolve" or "dual" (default: san)""")

parser.add_argument('--glimpse', type=int, default=2,
                    help='glimpse in Bilinear Attention Networks')

parser.add_argument('--num_stacks', type=int, default=2,
                    help='Number of Stacks in Stacked Attention Networks')
parser.add_argument('--drop_ratio', type=float, default=0.5,
                    help='Dropout ratio in Stacked Attention Networks')
parser.add_argument('--num_text_candidates', type=int, default=2,
                    help='Number of Text Candidates in Stacked Attention Networks')

parser.add_argument('--num_hidden', type=int, default=256,
                    help='Number of Hidden size in Both Attention Networks')

#AE arguments
parser.add_argument('--auto-encoder', action='store_true',
                    help='use AE or not')
parser.add_argument('--ae-model-path', type=str, default="./ae/ae_full_prelu.pth",
                    help='pretrained AE model')

args = parser.parse_args()

if args.evaluate == 0:
    args.use_train_instructions = 1
    log_filename = "train.log"
elif args.evaluate == 1:
    args.use_train_instructions = 1
    args.num_processes = 0
    log_filename = "test-MT.log"
elif args.evaluate == 2:
    args.use_train_instructions = 0
    args.num_processes = 0
    log_filename = "test-ZSL.log"
else:
    assert False, "Invalid evaluation type"

device = 'cpu'
print("I am using ", device)

def pertubation_map():
    env = GroundingEnv(args)

    env.game_init()
    (image, depth, instruction), _, _ = env.reset()
    original_image = image
    original_image = np.moveaxis(original_image, 0, -1)
    print(instruction)
    instruction_idx = []
    for word in instruction.split(" "):
        instruction_idx.append(env.word_to_idx[word])
    instruction_idx = np.array(instruction_idx).astype(float)
    image = torch.from_numpy(image).float()/255.0
    instruction_idx = torch.from_numpy(instruction_idx).view(1, -1).float()
    instruction_idx = instruction_idx.float().to(device)

    args.vocab_size = len(env.word_to_idx)
    args.dictionary = env.word_to_idx
    # with open("dict.txt", "w") as f:
    #     f.write("{}\n".format(args.dictionary))


    #load model
    model = A3C_LSTM_GA(args, ae_model=None).to(device)
    model.load_state_dict(torch.load('/home/tinvn/TIN/NLP_RL_Code/AE_VLN_Vizdoom/saved/convolve/train_easy_convolve', map_location=torch.device(device)))
    model = model.to(device)

    print("get a rollout of the policy...")
    history = rollout(model, env, max_ep_len=3e3)

    f = plt.figure(figsize=[30, 50])
    frame_ix = 4
    plt.imshow(history['ins'][frame_ix].moveaxis(0, 2))
    for a in f.axes: a.get_xaxis().set_visible(False) ; a.get_yaxis().set_visible(False)
    plt.show()
    plt.close()

    # Get perturbation saliency map
    radius = 3
    density = 3

    actor_saliency = score_frame(model, history, frame_ix, radius, density, interp_func=occlude, mode='actor')
    critic_saliency = score_frame(model, history, frame_ix, radius, density, interp_func=occlude, mode='critic')
    print(actor_saliency)
    print(critic_saliency)
    # upsample perturbation saliencies
    frame = history['ins'][frame_ix].moveaxis(0, 2).cpu().detach().numpy()
    frame = saliency_on_vizdoom_frame(actor_saliency, frame, fudge_factor=200, channel=2)
    perturbation_map = saliency_on_vizdoom_frame(critic_saliency, frame, fudge_factor=100, channel=0)

    f = plt.figure(figsize=[30, 50])
    plt.imshow(perturbation_map)
    plt.title('Pertubation map', fontsize=30)
    for a in f.axes: a.get_xaxis().set_visible(False) ; a.get_yaxis().set_visible(False)
    plt.show()

pertubation_map()