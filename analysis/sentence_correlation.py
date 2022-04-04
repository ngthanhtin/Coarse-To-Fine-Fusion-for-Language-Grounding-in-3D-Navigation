import sys
sys.path.insert(0, './')
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

import plotly
import plotly.graph_objs as go
from matplotlib import cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import jellyfish
import os, cv2, io
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image

import torch
from torch import nn
import torch.nn.functional as F

from models.models_fft import A3C_LSTM_GA
import argparse
from env.env import GroundingEnv
import pandas as pd

def _plot_helper(arr, xlabel, ylabel):
    plt.plot(arr, lw=2.0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.show()

def show_features_map(features):
    """
    feature map: (B, C, W, H)
    """
    feature_map = features.squeeze(0).detach()
    gray_scale = torch.sum(feature_map,0)
    gray_scale = gray_scale / feature_map.shape[0]
    
    fig = plt.figure(figsize=(30, 50))
    a = fig.add_subplot(1, 1, 1)
    imgplot = plt.imshow(gray_scale)
    a.axis("off")
    a.set_title('aaa', fontsize=30)
    # plt.savefig(str('feature_maps.jpg'), bbox_inches='tight')
    plt.show()

# define a function which returns an image as numpy array from figure
def get_img_from_fig(fig, dpi=60):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    return img

parser = argparse.ArgumentParser(description='Instruction Navigation in Vizdoom')

# Environment arguments
parser.add_argument('-env', '--env-type', type=str, default="single",
                    help="""Type of the environment,
                    "single", "twogoal" or "threegoal" (default: single)""")
parser.add_argument('-l', '--max-episode-length', type=int, default=60,
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
                    # default="data/multigoal_instruction/twogoal_instructions_train.json",
                    default="data/instructions_train.json",
                    help="""Train instructions file
                    (default: data/instructions_train.json)""")
parser.add_argument('--test-instr-file', type=str,
                    # default="data/multigoal_instruction/twogoal_instructions_test.json",
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
parser.add_argument('-att', '--attention', type=str, default="fga",
                    help="""Type of attention,
                    "fga" or "ga" (default: fga)""")
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

if args.env_type == 'single':
    args.max_episode_length = 30

if args.env_type == 'single':
    args.train_instr_file = "data/instructions_train.json"
    args.test_instr_file = "data/instructions_test.json"
    

device = 'cpu'
print("I am using ", device)

# ---------------------------------------------------- #

def GetSentenceEmbedding_CA():
    list_sentences = []
    dict_sentences_embedding = {}

    if args.env_type == 'single':
        env = GroundingEnv(args)
        for component in env.train_instructions:
            list_sentences.append(component['instruction'])
    
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
    with open("dict.txt", "w") as f:
        f.write("{}\n".format(args.dictionary))


    #load model
    model = A3C_LSTM_GA(args, ae_model=None).to(device)
    # model.load_state_dict(torch.load('/home/tinvn/TIN/NLP_RL_Code/AE_VLN_Vizdoom/saved/convolve/train_easy_convolve', map_location=torch.device(device)))
    model = model.eval()
    model = model.to(device)

    # Split model in two parts
    model_list = list(model.children())

    image_fn = nn.Sequential(model_list[6], model_list[0], model_list[7], model_list[0], model_list[8], model_list[0])
    text_fn = nn.Sequential(model_list[1], model_list[2])
    attn_linear_fn = model_list[3]
    sub_text_feat_0 = attn_linear_fn.layers[0]
    sub_text_feat_1 = attn_linear_fn.layers[1]
    sub_text_feat_2 = attn_linear_fn.layers[2]
    sub_text_feat_3 = attn_linear_fn.layers[3]
    sub_text_feat_4 = attn_linear_fn.layers[4]
    # print(attn_linear_fn.layers[0])

    outtro_cnn = nn.Sequential(model_list[4], model_list[0], model_list[5], model_list[0])
    
    linear_fn = nn.Sequential(model_list[10], model_list[0])
    
    time_emb_fn = model_list[9]
    hc_fn = model_list[11]
    actor_fn = model_list[13]
    critic_fn = model_list[12]

    episode_length = 0
    done = True
    
    num_episode = 0
    reward_sum = 0
    save_images_list = []

    image_feats_arr = []
    text_feats_arr = []
    fft_images_feats_arr = []
    fft_text_feats_arr = []

    save_path = './saved_images/'
    list_att_images = []

    sentence_emb_episode = []
    while True:
        episode_length += 1
        if done:
            episode_length = 0
            cx = torch.Tensor(torch.zeros(1, 256)).to(device)
            hx = torch.Tensor(torch.zeros(1, 256)).to(device)
        else:
            cx = torch.Tensor(cx.data).to(device)
            hx = torch.Tensor(hx.data).to(device)

        tx = torch.Tensor(torch.from_numpy(np.array([episode_length])).float()).to(device) #.long()
        instruction_idx = instruction_idx.float().to(device)

        # image = F.interpolate(image.unsqueeze(0), (224, 224))
        image = image.unsqueeze(0)
        image_feats = image_fn(torch.Tensor(image))
        # show_features_map(image_feats)
        
        text_feats = text_fn(torch.Tensor(instruction_idx).long())
        sentence_emb_episode.append(text_feats.squeeze().detach().cpu().numpy())

        # attention
        sub_text_ln_0 = sub_text_feat_0(text_feats).unsqueeze(2).unsqueeze(2)
        sub_text_ln_1 = sub_text_feat_1(text_feats).unsqueeze(2).unsqueeze(2)
        sub_text_ln_2 = sub_text_feat_2(text_feats).unsqueeze(2).unsqueeze(2)
        sub_text_ln_3 = sub_text_feat_3(text_feats).unsqueeze(2).unsqueeze(2)
        sub_text_ln_4 = sub_text_feat_4(text_feats).unsqueeze(2).unsqueeze(2)

        
        attention_0 = F.conv2d(image_feats, sub_text_ln_0, stride=(1,1))
        att_feats = attention_0
        attention_1 = F.conv2d(image_feats, sub_text_ln_1, stride=(1,1))
        att_feats = torch.cat([att_feats, attention_1], 0)
        attention_2 = F.conv2d(image_feats, sub_text_ln_2, stride=(1,1))
        att_feats = torch.cat([att_feats, attention_2], 0)
        attention_3 = F.conv2d(image_feats, sub_text_ln_3, stride=(1,1))
        att_feats = torch.cat([att_feats, attention_3], 0)
        attention_4 = F.conv2d(image_feats, sub_text_ln_4, stride=(1,1))
        att_feats = torch.cat([att_feats, attention_4], 0)

    
        att_feats = outtro_cnn(att_feats)
        att_feats = att_feats.view(-1).unsqueeze(0)
        
        #show cca
        acts1 = image_feats.clone().detach()
        # pool2 = attn_linear_fn(text_feats)
        pool2 = sub_text_ln_4
        pool2 = pool2.clone().detach()
        # pool2 = pool2.unsqueeze(2).unsqueeze(3)
        pool2 = pool2.expand(1, 64,12,12)
        acts1 = acts1.numpy()
        pool2 = pool2.numpy()
        
        fft_images_feats_arr.append(np.moveaxis(acts1, 1, -1))
        fft_text_feats_arr.append(np.moveaxis(pool2, 1, -1))

        avg_acts1 = np.mean(acts1, axis=(2,3))
        avg_pool2 = np.mean(pool2, axis=(2,3))
        image_feats_arr.append(avg_acts1)
        text_feats_arr.append(avg_pool2)
        # print(len(image_feats_arr))
        
        ##
        linear_feats = linear_fn(att_feats)
        hx_feats, cx_feats = hc_fn(linear_feats, (hx, cx))
        
        time_emb_feats = time_emb_fn(tx.long())
        x_feats = torch.cat((hx_feats, time_emb_feats.view(-1, 32)), 1)
        critic_feats = critic_fn(x_feats)
        actor_feats = actor_fn(x_feats)

        prob = F.softmax(actor_feats, dim=-1)
        action0 = prob.max(1)[1].unsqueeze(0).float()
        
        log_prob = F.log_softmax(actor_feats,dim=-1)
        action = prob.multinomial(num_samples=1).data.float()
        log_prob = log_prob.gather(1, torch.Tensor(action0).long())
        
        # env
        # image = F.interpolate(image, (224, 224))
        
        _, _, (_, _), att_img = model(
                (torch.Tensor(image),
                 torch.Tensor(instruction_idx), (tx, hx, cx)))
        list_att_images.append(att_img)
        # action = action.numpy()[0, 0]
        action = prob.max(1)[1].data.numpy()
        save_images_list.append(image)
        if args.env_type == 'single':
            (image, _, _), reward, done = env.step(action[0])
        else:
            (image, _, _), reward, done, _ = env.step(action[0])
        
        done = done or episode_length >= args.max_episode_length
        reward_sum += reward

        if done:
            print(len(dict_sentences_embedding.keys()))
            if instruction not in dict_sentences_embedding.keys():
                sentence_emb_episode = np.array(sentence_emb_episode)
                average_emb = np.mean(sentence_emb_episode, axis=0)
                sentence_emb_episode = []
                dict_sentences_embedding[instruction] = average_emb
            if len(dict_sentences_embedding.keys()) == len(list_sentences):
                np.save('no_ca_sentence.npy', dict_sentences_embedding)
                print("Finish")
                exit()

            image = torch.from_numpy(image).float()/255.0
            tx = torch.Tensor(torch.from_numpy(np.array([episode_length])).float()).to(device) #.long()
            instruction_idx = instruction_idx.float().to(device)

            # image = F.interpolate(image.unsqueeze(0), (224, 224))
            image = image.unsqueeze(0)
            image_feats = image_fn(torch.Tensor(image))
            text_feats = text_fn(torch.Tensor(instruction_idx).long())
            
            # attention
            sub_text_ln_0 = sub_text_feat_0(text_feats).unsqueeze(2).unsqueeze(2)
            sub_text_ln_1 = sub_text_feat_1(text_feats).unsqueeze(2).unsqueeze(2)
            sub_text_ln_2 = sub_text_feat_2(text_feats).unsqueeze(2).unsqueeze(2)
            sub_text_ln_3 = sub_text_feat_3(text_feats).unsqueeze(2).unsqueeze(2)
            sub_text_ln_4 = sub_text_feat_4(text_feats).unsqueeze(2).unsqueeze(2)

            
            attention_0 = F.conv2d(image_feats, sub_text_ln_0, stride=(1,1))
            att_feats = attention_0
            attention_1 = F.conv2d(image_feats, sub_text_ln_1, stride=(1,1))
            att_feats = torch.cat([att_feats, attention_1], 0)
            attention_2 = F.conv2d(image_feats, sub_text_ln_2, stride=(1,1))
            att_feats = torch.cat([att_feats, attention_2], 0)
            attention_3 = F.conv2d(image_feats, sub_text_ln_3, stride=(1,1))
            att_feats = torch.cat([att_feats, attention_3], 0)
            attention_4 = F.conv2d(image_feats, sub_text_ln_4, stride=(1,1))
            att_feats = torch.cat([att_feats, attention_4], 0)

        
            att_feats = outtro_cnn(att_feats)
            att_feats = att_feats.view(-1).unsqueeze(0)

            linear_feats = linear_fn(att_feats)
            hx_feats, cx_feats = hc_fn(linear_feats, (hx, cx))
            time_emb_feats = time_emb_fn(tx.long())
            x_feats = torch.cat((hx_feats, time_emb_feats.view(-1, 32)), 1)
            critic_feats = critic_fn(x_feats)
            actor_feats = actor_fn(x_feats)

            prob = F.softmax(actor_feats, dim=-1)
            action0 = prob.max(1)[1].unsqueeze(0).float()
            
            log_prob = F.log_softmax(actor_feats,dim=-1)
            action = prob.multinomial(num_samples=1).data.float()
            log_prob = log_prob.gather(1, torch.Tensor(action0).long())
      

            # if reward_sum == 2.: # for two goal setting
            #     # save image
            #     # root_path = './docs/videos/' + str(num_episode) + '_' + instruction
            #     root_path = save_path + str(num_episode) + '_' + instruction
            #     if not os.path.isdir(root_path):
            #         os.makedirs(root_path)
            #     for image_index in range(len(save_images_list)):
            #         cv2.imwrite(root_path + "/{}.png".format(image_index), save_images_list[image_index])
            #         cv2.imwrite(root_path + "/att_{}.png".format(image_index), save_images_list[image_index])
            # print(reward_sum)
            reward_sum = 0
            save_images_list = []
            list_att_images = []
            num_episode += 1
            
            
            (image, depth, instruction), _, _ = env.reset()
            print(instruction)

            instruction_idx = []
            for word in instruction.split(" "):
                instruction_idx.append(env.word_to_idx[word])
            instruction_idx = np.array(instruction_idx)
            instruction_idx = torch.from_numpy(instruction_idx).view(1, -1)

        original_image = image
        original_image = np.moveaxis(original_image, 0, -1)
        image = torch.from_numpy(image).float()/255.0

def GetSentenceEmbedding_FFT():
    list_sentences = []
    dict_sentences_embedding = {}

    if args.env_type == 'single':
        env = GroundingEnv(args)
        for component in env.train_instructions:
            list_sentences.append(component['instruction'])

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
    with open("dict.txt", "w") as f:
        f.write("{}\n".format(args.dictionary))


    #load model
    model = A3C_LSTM_GA(args, ae_model=None).to(device)
    model.load_state_dict(torch.load('/home/tinvn/TIN/NLP_RL_Code/FFT_VLN_Vizdoom/saved/fourier_models/single_goal/easy/train_easy_fourier_d1', map_location=torch.device(device)))
    # model.load_state_dict(torch.load('/home/tinvn/TIN/NLP_RL_Code/DeepRL-Grounding/saved/fourier_models/two_goals/easy/train_easy_multigoal_fourier_d1', map_location=torch.device(device)))
    # model.load_state_dict(torch.load('/home/tinvn/TIN/NLP_RL_Code/DeepRL-Grounding/saved/fourier_models/three_goals/easy/train_easy_threegoal_fourier_d1', map_location=torch.device(device)))
    model = model.eval()
    model = model.to(device)

    # Split model in two parts
    model_list = list(model.children())
    # print(model_list[3].ff)
    # print(model_list[3].layers)
    # exit()
    image_fn = nn.Sequential(model_list[4], model_list[0], model_list[5], model_list[0], model_list[6], model_list[0])
    text_fn = nn.Sequential(model_list[1], model_list[2])
    attn_linear_fn = model_list[3].attn_linear
    att_fn = model_list[3]

    fnet_blockimage = model_list[3].layers[0][0]
    fnet_blocktext = model_list[3].layers[0][1]
    linear_fn = nn.Sequential(model_list[8], model_list[0])
    time_emb_fn = model_list[7]
    hc_fn = model_list[9]
    actor_fn = model_list[11]
    critic_fn = model_list[10]

    episode_length = 0
    done = True
    
    num_episode = 0
    reward_sum = 0
    save_images_list = []

    image_feats_arr = []
    text_feats_arr = []
    fft_images_feats_arr = []
    fft_text_feats_arr = []

    sentence_emb_episode = []
    while True:
        episode_length += 1
        if done:
            episode_length = 0
            cx = torch.Tensor(torch.zeros(1, 256)).to(device)
            hx = torch.Tensor(torch.zeros(1, 256)).to(device)
        else:
            cx = torch.Tensor(cx.data).to(device)
            hx = torch.Tensor(hx.data).to(device)

        tx = torch.Tensor(torch.from_numpy(np.array([episode_length])).float()).to(device) #.long()
        instruction_idx = instruction_idx.float().to(device)

        image = F.interpolate(image.unsqueeze(0), (224, 224))
        image_feats = image_fn(torch.Tensor(image))
        # show_features_map(image_feats)
        # print(image_feats.shape)
        text_feats = text_fn(torch.Tensor(instruction_idx).long())
        sentence_emb_episode.append(text_feats.squeeze().detach().cpu().numpy())

        fnet_blockimage_feats = fnet_blockimage(image_feats)
        # show_features_map(fnet_blockimage_feats)
        fnet_blocktext_feats = fnet_blocktext(text_feats)
        att_feats = att_fn(image_feats, text_feats)
        # show_features_map(att_feats.reshape(1, 64, 12, 12))

        #show cca
        acts1 = image_feats.clone().detach()
        pool2 = attn_linear_fn(text_feats)
        pool2 = pool2.clone().detach()
        pool2 = pool2.unsqueeze(2).unsqueeze(3)
        pool2 = pool2.expand(1, 64,12,12)
        acts1 = acts1.numpy()
        pool2 = pool2.numpy()
        
        fft_images_feats_arr.append(np.moveaxis(acts1, 1, -1))
        fft_text_feats_arr.append(np.moveaxis(pool2, 1, -1))

        avg_acts1 = np.mean(acts1, axis=(2,3))
        avg_pool2 = np.mean(pool2, axis=(2,3))
        image_feats_arr.append(avg_acts1)
        text_feats_arr.append(avg_pool2)
        # print(len(image_feats_arr))
        
        linear_feats = linear_fn(att_feats)
        hx_feats, cx_feats = hc_fn(linear_feats, (hx, cx))
        time_emb_feats = time_emb_fn(tx.long())
        x_feats = torch.cat((hx_feats, time_emb_feats.view(-1, 32)), 1)
        critic_feats = critic_fn(x_feats)
        actor_feats = actor_fn(x_feats)

        prob = F.softmax(actor_feats, dim=-1)
        action0 = prob.max(1)[1].unsqueeze(0).float()
        
        log_prob = F.log_softmax(actor_feats,dim=-1)
        action = prob.multinomial(num_samples=1).data.float()
        log_prob = log_prob.gather(1, torch.Tensor(action0).long())
        
        # att_grads = torch.autograd.grad(critic_feats + log_prob, att_feats[2], retain_graph=True)
        image_grads = torch.autograd.grad(log_prob, image_feats)
        
        # image_grads = image_grads[0].cpu().data.numpy()
   
        w = image_grads[0][0].mean(-1).mean(-1)
    
        # image_feats = torch.fft.fft(torch.fft.fft(image_feats, dim=-1), dim=-2).real
        sal = torch.matmul(w, image_feats.view(64, 12*12))
        sal = torch.nn.ReLU()(sal)
        # sal = torch.matmul(w, att_feats[2].reshape(64, 12*12))
       
        sal = sal.view(12, 12).cpu().detach().numpy()
        sal = np.maximum(sal, 0)

        img = Image.fromarray(original_image).resize((224,224), Image.ANTIALIAS)
        sal = Image.fromarray(sal)
        sal = sal.resize(img.size, resample=Image.LINEAR)

        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))
        ax1.set_title('Original')
        ax2.set_title('Grad Map')
        _ = ax1.imshow(img)
        _ = ax2.imshow(img)
        _ = ax2.imshow(np.array(sal), alpha=0.5, cmap='jet')
        
        plot_img_np = get_img_from_fig(fig)
        save_images_list.append(plot_img_np)
        # plt.show()
        plt.close()
     
        
        # env
        # action = action.numpy()[0, 0]
        action = prob.max(1)[1].data.numpy()

        if args.env_type == 'single':
            (image, _, _), reward, done = env.step(action[0])
        else:
            (image, _, _), reward, done, _ = env.step(action[0])
            
        done = done or episode_length >= args.max_episode_length
        reward_sum += reward

        if done:
            print(len(dict_sentences_embedding.keys()))
            if instruction not in dict_sentences_embedding.keys():
                sentence_emb_episode = np.array(sentence_emb_episode)
                average_emb = np.mean(sentence_emb_episode, axis=0)
                sentence_emb_episode = []
                dict_sentences_embedding[instruction] = average_emb
            if len(dict_sentences_embedding.keys()) == len(list_sentences):
                np.save('fft_sentence.npy', dict_sentences_embedding)
                print("Finish")
                exit()

            image = torch.from_numpy(image).float()/255.0
            tx = torch.Tensor(torch.from_numpy(np.array([episode_length])).float()).to(device) #.long()
            instruction_idx = instruction_idx.float().to(device)

            image = F.interpolate(image.unsqueeze(0), (224, 224))
            image_feats = image_fn(torch.Tensor(image))
            text_feats = text_fn(torch.Tensor(instruction_idx).long())
            fnet_blockimage_feats = fnet_blockimage(image_feats)
            fnet_blocktext_feats = fnet_blocktext(text_feats)
            att_feats = att_fn(image_feats, text_feats)
            linear_feats = linear_fn(att_feats)
            hx_feats, cx_feats = hc_fn(linear_feats, (hx, cx))
            time_emb_feats = time_emb_fn(tx.long())
            x_feats = torch.cat((hx_feats, time_emb_feats.view(-1, 32)), 1)
            critic_feats = critic_fn(x_feats)
            actor_feats = actor_fn(x_feats)

            prob = F.softmax(actor_feats, dim=-1)
            action0 = prob.max(1)[1].unsqueeze(0).float()
            
            log_prob = F.log_softmax(actor_feats,dim=-1)
            action = prob.multinomial(num_samples=1).data.float()
            log_prob = log_prob.gather(1, torch.Tensor(action0).long())
            
            # att_grads = torch.autograd.grad(critic_feats + log_prob, att_feats[2], retain_graph=True)

            image_grads = torch.autograd.grad(log_prob, image_feats)

            w = image_grads[0][0].mean(-1).mean(-1)
            # image_feats = image_feats[-1]
            # image_feats = torch.fft.fft(torch.fft.fft(image_feats, dim=-1), dim=-2).real
            sal = torch.matmul(w, image_feats.view(64, 12*12))
            sal = torch.nn.ReLU()(sal)
            # sal = torch.matmul(w, att_feats[2].reshape(64, 12*12))
        
            sal = sal.view(12, 12).cpu().detach().numpy()
            sal = np.maximum(sal, 0)
    
            img = Image.fromarray(original_image).resize((224,224), Image.ANTIALIAS)
            sal = Image.fromarray(sal)
            sal = sal.resize(img.size, resample=Image.LINEAR)

            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))
            ax1.set_title('Original')
            ax2.set_title('Grad Map')
            _ = ax1.imshow(img)
            _ = ax2.imshow(img)
            _ = ax2.imshow(np.array(sal), alpha=0.5, cmap='jet')
            
            plot_img_np = get_img_from_fig(fig)
            save_images_list.append(plot_img_np)
            # plt.show()
            plt.close()
      

            # if reward_sum == 2.: # for two goal setting
            #     # save image
            #     root_path = './docs/videos/' + str(num_episode) + '_' + instruction
            #     if not os.path.isdir(root_path):
            #         os.makedirs(root_path)
            #     for image_index in range(len(save_images_list)):
            #         cv2.imwrite(root_path + "/{}.png".format(image_index), save_images_list[image_index])

            reward_sum = 0
            save_images_list = []
            num_episode += 1
            
            # print(reward)
            (image, depth, instruction), _, _ = env.reset()
            print(instruction)

            instruction_idx = []
            for word in instruction.split(" "):
                instruction_idx.append(env.word_to_idx[word])
            instruction_idx = np.array(instruction_idx)
            instruction_idx = torch.from_numpy(instruction_idx).view(1, -1)

        original_image = image
        original_image = np.moveaxis(original_image, 0, -1)
        image = torch.from_numpy(image).float()/255.0
        


def show_correlation_matrix(np_file_path):

    fft_emb_file = np.load(np_file_path, allow_pickle=True)
    fft_emb_file = fft_emb_file.item()

    embs_number = []
    sentences = []
    for key in fft_emb_file.keys():
        sentences.append(key)
        embs_number.append(fft_emb_file[key])
        
    # print(len(embs_number))
    # print(len(embs_number[0]))

    trim_embs = []
    trim_sentences = []

    start_string = sentences[0]
    trim_embs.append(embs_number[0])
    trim_sentences.append(sentences[0])
    for i, (emb, sent) in enumerate(zip(embs_number, sentences)):
        if sent == start_string:
            continue
        else:
            start_string = sent
            trim_embs.append(embs_number[i])
            trim_sentences.append(start_string)

    # remove duplicate
    non_dup_embs = []
    non_dup_sentences = []

    sent_2_emb = {}
    for i, (sent, emb) in enumerate(zip(trim_sentences, trim_embs)):
        if sent not in sent_2_emb:
                sent_2_emb[sent] = emb

    for sent in sent_2_emb:
        non_dup_sentences.append(sent)
        non_dup_embs.append(sent_2_emb[sent])


    for i, emb in enumerate(non_dup_embs):
        max_value = max(emb)
        non_dup_embs[i] = [k/max_value for k in emb]

    # sort by object, armor, pillar,torch, keycard, skullkey
    objects_sentences = []
    objects_embs = []
    armors_sentences = []
    armors_embs = []
    pillars_sentences = []
    pillars_embs = []
    torchs_sentences = []
    torchs_embs = []
    keycards_sentences = []
    keycards_embs = []
    skullkey_sentences = []
    skullkey_embs = []

    for i, (sent, emb) in enumerate(zip(non_dup_sentences, non_dup_embs)):
        if 'object' in sent:
            objects_sentences.append(sent)
            objects_embs.append(emb)
        elif 'armor' in sent:
            armors_sentences.append(sent)
            armors_embs.append(emb)
        elif 'pillar' in sent:
            pillars_sentences.append(sent)
            pillars_embs.append(emb)
        elif 'torch' in sent:
            torchs_sentences.append(sent)
            torchs_embs.append(emb)
        elif 'keycard' in sent:
            keycards_sentences.append(sent)
            keycards_embs.append(emb)
        elif 'skullkey' in sent:
            skullkey_sentences.append(sent)
            skullkey_embs.append(emb)

    new_sentences = []
    new_embs = []
    for sent, emb in zip(objects_sentences, objects_embs):
        new_sentences.append(sent)
        new_embs.append(emb)
    for sent, emb in zip(armors_sentences, armors_embs):
        new_sentences.append(sent)
        new_embs.append(emb)
    for sent, emb in zip(pillars_sentences, pillars_embs):
        new_sentences.append(sent)
        new_embs.append(emb)
    for sent, emb in zip(torchs_sentences, torchs_embs):
        new_sentences.append(sent)
        new_embs.append(emb)
    for sent, emb in zip(keycards_sentences, keycards_embs):
        new_sentences.append(sent)
        new_embs.append(emb)
    for sent, emb in zip(skullkey_sentences, skullkey_embs):
        new_sentences.append(sent)
        new_embs.append(emb)

    def cosine(u, v):
        cosine_ = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
        return cosine_ if cosine_ > 0 else cosine_ * (-1)
    def levenstein(u, v):
        return jellyfish.levenshtein_distance(u, v)

    def histogram_intersection(a, b):
        v = np.minimum(a, b).sum().round(decimals=1)
        return v
    df = pd.DataFrame()

    for i in range(len(new_sentences)):
        df.insert(i, new_sentences[i], new_embs[i], True)

    corr = df.corr(method='pearson')
    c2 = corr.copy()
    c2.values[np.tril_indices_from(c2)] = np.nan
    print(c2.unstack().mean())
    print(c2.unstack().std())
    plt.matshow(corr)
    plt.show()

    return
    sim_matrix = np.zeros((len(new_sentences), len(new_sentences)))
    sum_cos = 0.
    for i in range(len(new_sentences)):
        for j in range(len(new_sentences)):
            u = np.asarray(new_embs[i])
            v = np.asarray(new_embs[j])
            sim_matrix[i, j] = cosine(u, v)
            sim_matrix[j, i] = cosine(u, v)
            sum_cos += cosine(u, v)
            # sim_matrix[i, j] = levenstein(new_sentences[i], new_sentences[j])

    sum_cos = sum_cos/(i*j)
    print(sum_cos)
    labels = []
    for sent in new_sentences:
        appended_sent = sent.replace("Go to the ", "")
        labels.append(appended_sent)

    fig, ax = plt.subplots(figsize=(30,30))
    cax = ax.matshow(sim_matrix, interpolation='nearest')
    ax.grid(True)

    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    plt.title('Instruction Vector Cosine Similarity matrix')
    plt.xticks(range(len(new_sentences)), labels, rotation=90, fontsize=7)
    plt.yticks(range(len(new_sentences)), labels, rotation=0, fontsize=7)
    plt.gcf().subplots_adjust(bottom=0.2)
    fig.colorbar(cax, ticks=[-0.39, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1])
    # fig.colorbar(cax, ticks=[0.1, 2, 3, 4, 0.5, 0.6, 0.7, .75,.8,.85,.90,.95,17][::-1])
    plt.show()

# GetSentenceEmbedding_CA()
# GetSentenceEmbedding_FFT()
np_file_path = './ca_sentence.npy'
# np_file_path = './no_ca_sentence.npy'
# np_file_path = './fft_sentence.npy'
# np_file_path = './no_fft_sentence.npy'
show_correlation_matrix(np_file_path)