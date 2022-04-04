import sys
sys.path.insert(0, './')
import os, cv2, io
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image

import torch
from torch import nn
import torch.nn.functional as F

from models.models import A3C_LSTM_GA
from ae.auto_encoder import Auto_Encoder_Model_PReLu
import argparse
from env.env import GroundingEnv
import pandas as pd
from analysis.svcca import cca_core
from svcca.dft_ccas import fourier_ccas, fourier_ccas2
# from svcca.dft_ccas2 import fourier_ccas

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
else:
    args.max_episode_length = 60

if args.env_type == 'single':
    args.train_instr_file = "data/instructions_train.json"
    args.test_instr_file = "data/instructions_test.json"

device = 'cpu'
print("I am using ", device)

# ---------------------------------------------------- #

def GradCAM():
    if args.env_type == 'single':
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
    with open("dict.txt", "w") as f:
        f.write("{}\n".format(args.dictionary))


    #load model
    model = A3C_LSTM_GA(args, ae_model=None).to(device)
    model.load_state_dict(torch.load('./saved/cf_convolve/train_easy_convolve_cf', map_location=torch.device(device)))
    # model.load_state_dict(torch.load('./saved/cf_convolve/train_medium_cfconvolve', map_location=torch.device(device)))
    # model.load_state_dict(torch.load('./saved/train_hard_cfconvolve_continue', map_location=torch.device(device)))
    model = model.eval()
    model = model.to(device)

    # Split model in two parts
    model_list = list(model.children())
    
    image_fn_1 = nn.Sequential(model_list[1], model_list[0])
    image_fn_2 = nn.Sequential(model_list[2], model_list[0])
    image_fn_3 = nn.Sequential(model_list[3], model_list[0])
    text_fn = nn.Sequential(model_list[4], model_list[5])

    # cfca attention 
    channels = [128,64,64]
    layers = nn.ModuleList()
    attn_linear_fn = model_list[6]
    for k in range(5):
        for i in range(len(channels)):
            layers.append(attn_linear_fn.layers[k*len(channels) + i])

    outtro_cnn = nn.Sequential(model_list[7], model_list[0], model_list[8], model_list[0])
    
    linear_fn = nn.Sequential(model_list[10], model_list[0])
    
    time_emb_fn = model_list[9]
    hc_fn = model_list[11]
    critic_fn = model_list[12]
    actor_fn = model_list[13]
    

    episode_length = 0
    done = True
    
    num_episode = 0
    reward_sum = 0
    save_images_list = []

    image_feats_arr = []
    text_feats_arr = []
    cfca_images_feats_arr_1 = []
    cfca_images_feats_arr_2 = []
    cfca_images_feats_arr_3 = []
    cfca_text_feats_arr_1 = []
    cfca_text_feats_arr_2 = []
    cfca_text_feats_arr_3 = []

    save_path = './saved_images/'
    list_att_images = []

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
        img_feats = []
        image_feats_1 = image_fn_1(torch.Tensor(image))
        image_feats_2 = image_fn_2(image_feats_1)
        image_feats_3 = image_fn_3(image_feats_2)
        img_feats.append(image_feats_1)
        img_feats.append(image_feats_2)
        img_feats.append(image_feats_3)
        text_feats = text_fn(torch.Tensor(instruction_idx).long())
        
        # attention
         # resize convolution layers
        new_img_feats = []
        for img_feat in img_feats:
            new_img_feat = F.interpolate(img_feat, (17, 8))
            new_img_feats.append(new_img_feat)

        attention_maps = None
        for k in range(5):
            attention = torch.zeros(new_img_feats[0].size(0), 1, new_img_feats[0].size(2), new_img_feats[0].size(3))
            for i in range(len(channels)):
                sent_fc = layers[i + k*len(channels)](text_feats)
                sent_fc = sent_fc.unsqueeze(2).unsqueeze(2) # B, Dim, 1, 1
                attention += F.conv2d(new_img_feats[i], sent_fc, stride=(1,1))
                
            if k == 0:
                attention_maps = attention
            else:
                attention_maps = torch.cat([attention_maps, attention], 0)

        att_feats = outtro_cnn(attention_maps)
        att_feats = att_feats.view(-1).unsqueeze(0)
        
        #show cca
        acts1 = new_img_feats[0].clone().detach()
        acts2 = new_img_feats[1].clone().detach()
        acts3 = new_img_feats[2].clone().detach()
        
        pool2_1 = layers[12](text_feats) + layers[9](text_feats) + layers[6](text_feats) + layers[3](text_feats) + layers[0](text_feats)
        pool2_2 = layers[13](text_feats) + layers[10](text_feats) + layers[7](text_feats) + layers[4](text_feats) + layers[1](text_feats)
        pool2_3 = layers[14](text_feats) + layers[11](text_feats) + layers[8](text_feats) + layers[5](text_feats) + layers[2](text_feats)
        pool2_1 = pool2_1.clone().detach()
        pool2_2 = pool2_2.clone().detach()
        pool2_3 = pool2_3.clone().detach()
        pool2_1 = pool2_1.unsqueeze(2).unsqueeze(3)
        pool2_2 = pool2_2.unsqueeze(2).unsqueeze(3)
        pool2_3 = pool2_3.unsqueeze(2).unsqueeze(3)
        pool2_1 = pool2_1.expand(1, 128,12,12)
        pool2_2 = pool2_2.expand(1, 64,12,12)
        pool2_3 = pool2_3.expand(1, 64,12,12)
        acts1 = acts1.numpy()
        acts2 = acts2.numpy()
        acts3 = acts3.numpy()
        pool2_1 = pool2_1.numpy()
        pool2_2 = pool2_2.numpy()
        pool2_3 = pool2_3.numpy()
        
        cfca_images_feats_arr_1.append(np.moveaxis(acts1, 1, -1))
        cfca_images_feats_arr_2.append(np.moveaxis(acts2, 1, -1))
        cfca_images_feats_arr_3.append(np.moveaxis(acts3, 1, -1))
        cfca_text_feats_arr_1.append(np.moveaxis(pool2_1, 1, -1))
        cfca_text_feats_arr_2.append(np.moveaxis(pool2_2, 1, -1))
        cfca_text_feats_arr_3.append(np.moveaxis(pool2_3, 1, -1))


        # avg_acts1 = np.mean(acts1, axis=(2,3))
        # avg_acts2 = np.mean(acts2, axis=(2,3))
        # avg_acts3 = np.mean(acts3, axis=(2,3))
        # avg_pool2 = np.mean(pool2, axis=(2,3))
        #
        # image_feats_arr.append(avg_acts3)
        # text_feats_arr.append(avg_pool2)
        # print(len(image_feats_arr))
        
        dft = False
        # if len(image_feats_arr) == 1000 and dft == False:
        #     image_feats_arr = np.squeeze(np.asarray(image_feats_arr))
        #     text_feats_arr = np.squeeze(np.asarray(text_feats_arr))
        #     print("shapes after average pool over spatial dimensions", image_feats_arr.shape, text_feats_arr.shape)
        #     a_results = cca_core.get_cca_similarity(image_feats_arr.T, text_feats_arr.T, epsilon=1e-7,compute_dirns=True, verbose=True)
            
        #     all_results = pd.DataFrame()
        #     all_results = all_results.append(a_results, ignore_index=True)
        #     all_results.to_pickle('notrained.df')
        #     _plot_helper(a_results["cca_coef1"], "CCA Coef idx", "CCA coef value")
        #     print('{:.4f}'.format(np.mean(a_results["cca_coef1"][:20]))) 
            # normal pretrained(full, top20): 0.1476, 0.619 # normal w/o pretrained: 0.1436, 0.4864
            # fnet pretrained 0.5359# fnet without pretrained: 0.3199
            # 1000: 0.3878, 1000: 0.3089, 1000: no pretrained no fourier: 0.3732, no fourier with pretrained: 0.5885 (1e-7)
            # image_feats_arr = []
            # text_feats_arr = []
        if len(cfca_images_feats_arr_3) == 1000 and dft == True:
            cfca_images_feats_arr_1 = np.squeeze(np.asarray(cfca_images_feats_arr_1))
            cfca_images_feats_arr_2 = np.squeeze(np.asarray(cfca_images_feats_arr_2))
            cfca_images_feats_arr_3 = np.squeeze(np.asarray(cfca_images_feats_arr_3))
            cfca_text_feats_arr_1 = np.squeeze(np.asarray(cfca_text_feats_arr_1))
            cfca_text_feats_arr_2 = np.squeeze(np.asarray(cfca_text_feats_arr_2))
            cfca_text_feats_arr_3 = np.squeeze(np.asarray(cfca_text_feats_arr_3))
            # print(cfca_images_feats_arr.shape, cfca_text_feats_arr.shape)
            result_1 = fourier_ccas2(cfca_images_feats_arr_1, cfca_text_feats_arr_1, return_coefs=True, compute_dirns=True)
            result_2 = fourier_ccas2(cfca_images_feats_arr_2, cfca_text_feats_arr_2, return_coefs=True, compute_dirns=True)
            result_3 = fourier_ccas2(cfca_images_feats_arr_3, cfca_text_feats_arr_3, return_coefs=True, compute_dirns=True)
            cfca_images_feats_arr_1 = []
            cfca_images_feats_arr_2 = []
            cfca_images_feats_arr_3 = []
            cfca_text_feats_arr = []
            result_1.to_pickle('../cfcaae_1.df')
            result_2.to_pickle('../cfcaae_2.df')
            result_3.to_pickle('../cfcaae_3.df')
            exit()
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
            
            image = torch.from_numpy(image).float()/255.0
            tx = torch.Tensor(torch.from_numpy(np.array([episode_length])).float()).to(device) #.long()
            instruction_idx = instruction_idx.float().to(device)

            # image = F.interpolate(image.unsqueeze(0), (224, 224))
            image = image.unsqueeze(0)
            img_feats = []
            image_feats_1 = image_fn_1(torch.Tensor(image))
            image_feats_2 = image_fn_2(image_feats_1)
            image_feats_3 = image_fn_3(image_feats_2)
            img_feats.append(image_feats_1)
            img_feats.append(image_feats_2)
            img_feats.append(image_feats_3)
            text_feats = text_fn(torch.Tensor(instruction_idx).long())
            
            # attention
            # resize convolution layers
            new_img_feats = []
            for img_feat in img_feats:
                new_img_feat = F.interpolate(img_feat, (17, 8))
                new_img_feats.append(new_img_feat)

            attention_maps = None
            for k in range(5):
                attention = torch.zeros(new_img_feats[0].size(0), 1, new_img_feats[0].size(2), new_img_feats[0].size(3))
                for i in range(len(channels)):
                    sent_fc = layers[i + k*len(channels)](text_feats)
                    sent_fc = sent_fc.unsqueeze(2).unsqueeze(2) # B, Dim, 1, 1
                    attention += F.conv2d(new_img_feats[i], sent_fc, stride=(1,1))
                    
                if k == 0:
                    attention_maps = attention
                else:
                    attention_maps = torch.cat([attention_maps, attention], 0)

            att_feats = outtro_cnn(attention_maps)
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
      
            # save image
            if reward_sum == 1.:
                root_path = save_path + str(num_episode) + '_' + instruction
                if not os.path.isdir(root_path):
                    os.makedirs(root_path)
                for image_index in range(len(save_images_list)):
                    ori_img = save_images_list[image_index].squeeze().permute(1,2,0)
                    ori_img = ori_img.detach().cpu().numpy()*255.
                    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(root_path + "/{}.png".format(image_index), ori_img)
                    cv2.imwrite(root_path + "/att_{}.png".format(image_index), list_att_images[image_index])
                    
                
            print(reward_sum)
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
        
GradCAM()


