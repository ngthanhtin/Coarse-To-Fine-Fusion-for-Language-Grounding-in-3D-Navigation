import numpy as np
from numpy.lib.shape_base import expand_dims
import torch
import torch.nn.functional as F
import time
import logging
import cv2
import os
import pandas as pd
import pickle
import multigoal_env as grounding_env
from new_models_2 import A3C_LSTM_GA
from ae.auto_encoder import Auto_Encoder_Model_PReLu224, Auto_Encoder_Model_PReLu
from constants import *
device='cpu'
log_file = 'train_easy_multigoal_fourier_d1_ae.log'




def save_weight_bias_gated(model, file_path):
    for name, param in model.named_parameters():
        if name == 'conv3.weight':
            x_emb_weight = param
        if name == 'conv3.bias':
            x_emb_bias = param
        if name == 's_emb.rnn.weight_ih_l0':
            s_emb_ih_weight = param
        if name == 's_emb.rnn.bias_ih_l0':
            s_emb_ih_bias = param
        if name == 's_emb.rnn.weight_hh_l0':
            s_emb_hh_weight = param
        if name == 's_emb.rnn.bias_hh_l0':
            s_emb_hh_bias = param
        if name == 'attention.attn_linear.weight': 
            attn_linear_weight = param
        if name == 'attention.attn_linear.bias':
            attn_linear_bias = param

    #save file
    if os.path.exists(file_path):
        #load file
        # name_dict = {
        #     'x_emb_weight': x_emb_weight.cpu().detach().numpy(),
        #     'x_emb_bias':x_emb_bias.cpu().detach().numpy(),
        #     's_emb_ih_weight':s_emb_ih_weight.cpu().detach().numpy(),
        #     's_emb_ih_bias':s_emb_ih_bias.cpu().detach().numpy(),
        #     's_emb_hh_weight':s_emb_hh_weight.cpu().detach().numpy(),
        #     's_emb_hh_bias':s_emb_hh_bias.cpu().detach().numpy(),
        #     'attn_linear_weight':attn_linear_weight.cpu().detach().numpy(),
        #     'attn_linear_bias':attn_linear_bias.cpu().detach().numpy()
        # }
        
        with open(file_path, 'rb') as handle:
            unserialized_data = pickle.load(handle)
            unserialized_data['x_emb_weight'].append(x_emb_weight.cpu().detach().numpy())
            unserialized_data['x_emb_bias'].append(x_emb_bias.cpu().detach().numpy())
            unserialized_data['s_emb_ih_weight'].append(s_emb_ih_weight.cpu().detach().numpy())
            unserialized_data['s_emb_ih_bias'].append(s_emb_ih_bias.cpu().detach().numpy())
            unserialized_data['s_emb_hh_weight'].append(s_emb_hh_weight.cpu().detach().numpy())
            unserialized_data['s_emb_hh_bias'].append(s_emb_hh_bias.cpu().detach().numpy())
            unserialized_data['attn_linear_weight'].append(attn_linear_weight.cpu().detach().numpy())
            unserialized_data['attn_linear_bias'].append(attn_linear_bias.cpu().detach().numpy())

        with open(file_path, 'wb') as handle:
            pickle.dump(unserialized_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        name_dict = {
            'x_emb_weight': [x_emb_weight.cpu().detach().numpy()],
            'x_emb_bias':[x_emb_bias.cpu().detach().numpy()],
            's_emb_ih_weight':[s_emb_ih_weight.cpu().detach().numpy()],
            's_emb_ih_bias':[s_emb_ih_bias.cpu().detach().numpy()],
            's_emb_hh_weight':[s_emb_hh_weight.cpu().detach().numpy()],
            's_emb_hh_bias':[s_emb_hh_bias.cpu().detach().numpy()],
            'attn_linear_weight':[attn_linear_weight.cpu().detach().numpy()],
            'attn_linear_bias':[attn_linear_bias.cpu().detach().numpy()]
        }

        with open(file_path, 'wb') as handle:
            pickle.dump(name_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_weight_bias(model, file_path):
    for name, param in model.named_parameters():
        if name == 'conv3.weight':
            x_emb_weight = param
        if name == 'conv3.bias':
            x_emb_bias = param
        if name == 's_emb.rnn.weight_ih_l0':
            s_emb_ih_weight = param
        if name == 's_emb.rnn.bias_ih_l0':
            s_emb_ih_bias = param
        if name == 's_emb.rnn.weight_hh_l0':
            s_emb_hh_weight = param
        if name == 's_emb.rnn.bias_hh_l0':
            s_emb_hh_bias = param
        # if name == 'attention.layers.0.0.norm.weight': 
        #     x_emb_norm_fourier_weight = param
        # if name == 'attention.layers.0.0.norm.bias':
        #     x_emb_norm_fourier_bias = param
        # if name == 'attention.layers.0.1.norm.weight':
        #     s_emb_norm_fourier_weight = param
        # if name == 'attention.layers.0.1.norm.bias':
        #     s_emb_norm_fourier_bias = param
        # if name == 'attention.ff.norm.weight':
        #     gated_attention_norm_weight = param
        # if name == 'attention.ff.norm.bias':
        #     gated_attention_norm_bias = param

    #save file
    if os.path.exists(file_path):
        #load file
        # name_dict = {
        #     'x_emb_weight': x_emb_weight.cpu().detach().numpy(),
        #     'x_emb_bias':x_emb_bias.cpu().detach().numpy(),
        #     's_emb_ih_weight':s_emb_ih_weight.cpu().detach().numpy(),
        #     's_emb_ih_bias':s_emb_ih_bias.cpu().detach().numpy(),
        #     's_emb_hh_weight':s_emb_hh_weight.cpu().detach().numpy(),
        #     's_emb_hh_bias':s_emb_hh_bias.cpu().detach().numpy(),
        #     'x_emb_norm_fourier_weight':x_emb_norm_fourier_weight.cpu().detach().numpy(),
        #     'x_emb_norm_fourier_bias':x_emb_norm_fourier_bias.cpu().detach().numpy(),
        #     's_emb_norm_fourier_weight':s_emb_norm_fourier_weight.cpu().detach().numpy(),
        #     's_emb_norm_fourier_bias':s_emb_norm_fourier_bias.cpu().detach().numpy(),
        #     'gated_attention_norm_weight':gated_attention_norm_weight.cpu().detach().numpy(),
        #     'gated_attention_norm_bias':gated_attention_norm_bias.cpu().detach().numpy()
        # }
        
        with open(file_path, 'rb') as handle:
            unserialized_data = pickle.load(handle)
            unserialized_data['x_emb_weights'].append(x_emb_weight.cpu().detach().numpy())
            unserialized_data['x_emb_bias'].append(x_emb_bias.cpu().detach().numpy())
            unserialized_data['s_emb_ih_weight'].append(s_emb_ih_weight.cpu().detach().numpy())
            unserialized_data['s_emb_ih_bias'].append(s_emb_ih_bias.cpu().detach().numpy())
            unserialized_data['s_emb_hh_weight'].append(s_emb_hh_weight.cpu().detach().numpy())
            unserialized_data['s_emb_hh_bias'].append(s_emb_hh_bias.cpu().detach().numpy())
            # unserialized_data['x_emb_norm_fourier_weight'].append(x_emb_norm_fourier_weight.cpu().detach().numpy())
            # unserialized_data['x_emb_norm_fourier_bias'].append(x_emb_norm_fourier_bias.cpu().detach().numpy())
            # unserialized_data['s_emb_norm_fourier_weight'].append(s_emb_norm_fourier_weight.cpu().detach().numpy())
            # unserialized_data['s_emb_norm_fourier_bias'].append(s_emb_norm_fourier_bias.cpu().detach().numpy())
            # unserialized_data['gated_attention_norm_weight'].append(gated_attention_norm_weight.cpu().detach().numpy())
            # unserialized_data['gated_attention_norm_bias'].append(gated_attention_norm_bias.cpu().detach().numpy())

        with open(file_path, 'wb') as handle:
            pickle.dump(unserialized_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        name_dict = {
            'x_emb_weight': [x_emb_weight.cpu().detach().numpy()],
            'x_emb_bias':[x_emb_bias.cpu().detach().numpy()],
            's_emb_ih_weight':[s_emb_ih_weight.cpu().detach().numpy()],
            's_emb_ih_bias':[s_emb_ih_bias.cpu().detach().numpy()],
            's_emb_hh_weight':[s_emb_hh_weight.cpu().detach().numpy()],
            's_emb_hh_bias':[s_emb_hh_bias.cpu().detach().numpy()],
            # 'x_emb_norm_fourier_weight':[x_emb_norm_fourier_weight.cpu().detach().numpy()],
            # 'x_emb_norm_fourier_bias':[x_emb_norm_fourier_bias.cpu().detach().numpy()],
            # 's_emb_norm_fourier_weight':[s_emb_norm_fourier_weight.cpu().detach().numpy()],
            # 's_emb_norm_fourier_bias':[s_emb_norm_fourier_bias.cpu().detach().numpy()],
            # 'gated_attention_norm_weight':[gated_attention_norm_weight.cpu().detach().numpy()],
            # 'gated_attention_norm_bias':[gated_attention_norm_bias.cpu().detach().numpy()]
        }
        
        with open(file_path, 'wb') as handle:
            pickle.dump(name_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # conv1 = nn.Conv2d(3, 1, 3)
    # weight = conv1.weight.data.numpy()
    # plt.imshow(weight[0, ...])



def test(rank, args, shared_model):

    torch.manual_seed(args.seed + rank)

    env = grounding_env.MultiGoal_GroundingEnv(args)
    env.game_init()

    ae_model = None
    if args.auto_encoder:
        ae_model = Auto_Encoder_Model_PReLu224()

    model = A3C_LSTM_GA(args, ae_model).to(device)
    if (args.load != "0"):
        print("Loading model ... "+args.load)
        model.load_state_dict(
            torch.load(args.load, map_location=lambda storage, loc: storage))
            
    model.eval()

    (image, depth, instruction), _, _= env.reset()
    # print(instruction)
    # depth =np.expand_dims(depth, axis=0)
    # image = np.concatenate((image, depth), axis=0)

    # Print instruction while evaluating and visualizing
    if args.evaluate != 0 and args.visualize == 1:
        print("Instruction: {} ".format(instruction))

    # Getting indices of the words in the instruction
    
    instruction_idx = []
    for word in instruction.split(" "):
        instruction_idx.append(env.word_to_idx[word])
    instruction_idx = np.array(instruction_idx).astype(float)
    original_image = image
    image = torch.from_numpy(image).float()/255.0
    instruction_idx = torch.from_numpy(instruction_idx).view(1, -1).float()

    reward_sum = 0
    done = True

    start_time = time.time()

    episode_length = 0
    reach_list = []
    rewards_list = []
    accuracy_list = []
    episode_length_list = []
    num_episode = 0
    best_reward = 0.0
    test_freq = 50
    
    #-------------BUGG-------------
    episode_count = 0
    image_index = 0
    #-----------------------------
    #------SAVE WEIGHTS, BIAS------#
    # root_path = './weight_bias/' + str(num_episode) + '_' + instruction
    # file_name = 'weight_bias.pickle'
    # if not os.path.isdir(root_path):
    #     os.makedirs(root_path)

    while True:
        episode_length += 1
        if done:
            if (args.evaluate == 0):
                model.load_state_dict(shared_model.state_dict())

            cx = torch.Tensor(torch.zeros(1, 256)).to(device)
            hx = torch.Tensor(torch.zeros(1, 256)).to(device)

            # root_path = './weight_bias/' + str(num_episode) + '_' + instruction
            # if not os.path.isdir(root_path):
            #     os.makedirs(root_path)
                

        else:
            cx = torch.Tensor(cx.data).to(device)
            hx = torch.Tensor(hx.data).to(device)

        tx = torch.Tensor(torch.from_numpy(np.array([episode_length])).float()).to(device) #.long()

        instruction_idx = instruction_idx.float().to(device)
        # print(instruction)
        # with open("word.txt", "a+") as f:
        #     f.write("{}\n".format(instruction))
        if args.auto_encoder:
            original_image = np.moveaxis(original_image, 0, 2)
            # cv2.imwrite('foo.png', cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))
            ae_input = original_image / 255.0
            
            ae_input = torch.Tensor(ae_input)
            ae_input = ae_input.permute(-1,0,1)
            ae_input = ae_input.unsqueeze(0)

            value, logit, (hx, cx), decoder, ae_input = model(
                (ae_input, torch.Tensor(image.unsqueeze(0)),
                 torch.Tensor(instruction_idx), (tx, hx, cx)))
        else:
            original_image = np.moveaxis(original_image, 0, 2)
            # cv2.imwrite('foo.png', cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))
            value, logit, (hx, cx) = model(
                    (torch.Tensor(image.unsqueeze(0)),
                    torch.Tensor(instruction_idx), (tx, hx, cx)))
            
            # if episode_length % 15 == 0: #20
            #     save_weight_bias_gated(model, root_path + '/' + file_name)  

        prob = F.softmax(logit, dim=-1)
        action = prob.max(1)[1].data.numpy()

        (image, depth, _), reward, done, (reach_1, reach_2) = env.step(action[0])

        #-------------BUGG-----------------------#
        # save image
        # tmp_image = np.moveaxis(image, 0, -1) # for rgb image
        
        # tmp_image = cv2.cvtColor(tmp_image, cv2.COLOR_BGR2RGB)
        # image_index += 1
        # MYDIR = '/home/tinvn/TIN/NLP_RL_Code/DeepRL-Grounding/data/multigoal_test_images/'+ str(episode_count)
        # import os
        # if not os.path.isdir(MYDIR):
        #     os.makedirs(MYDIR)
        #     print("created folder : ", MYDIR)
        # cv2.imwrite(MYDIR + '/{}.png'.format(image_index), tmp_image)
        #------------------------------------------------

        done = done or episode_length >= args.max_episode_length
        reward_sum += reward    

        if done:
            num_episode += 1 
            reach_list.append(reach_1+reach_2)
            rewards_list.append(reward_sum)
            # Print reward while evaluating and visualizing
            if args.evaluate != 0 and args.visualize == 1:
                print("Total reward: {}".format(reward_sum))

            episode_length_list.append(episode_length)
            # if reward == CORRECT_OBJECT_REWARD + CORRECT_OBJECT_REWARD/2:
            #     accuracy = 1
            # elif reward == CORRECT_OBJECT_REWARD:
            #     accuracy = 0.5
            # else:
            #     accuracy = 0

            if reach_1 + reach_2 == 2:
                accuracy = 1
                # save_weight_bias(model, root_path + '/' + file_name)  
            elif reach_1 + reach_2 == 1:
                accuracy = 0.5
                # save_weight_bias_gated(model, root_path + '/' + file_name)  
            else:
                accuracy = 0

            accuracy_list.append(accuracy)
            if(len(rewards_list) >= test_freq): 
                print(" ".join([
                    "Time {},".format(time.strftime("%Hh %Mm %Ss",
                                      time.gmtime(time.time() - start_time))),
                    "Avg Reward {},".format(np.mean(rewards_list)),
                    "Avg Accuracy {},".format(np.mean(accuracy_list)),
                    "Avg Ep length {},".format(np.mean(episode_length_list)),
                    "Avg Reach {},".format(np.mean(reach_list)),
                    "Best Reward {}".format(best_reward)]))
                with open(log_file, "a+") as f:
                    f.write(" ".join([
                        "Time {},".format(time.strftime("%Hh %Mm %Ss",
                                        time.gmtime(time.time() - start_time))),
                        "Avg Reward {},".format(np.mean(rewards_list)),
                        "Avg Accuracy {},".format(np.mean(accuracy_list)),
                        "Avg Ep length {},".format(np.mean(episode_length_list)),
                        "Avg Reach {},".format(np.mean(reach_list)),
                        "Best Reward {}\n".format(best_reward)]))
                if np.mean(rewards_list) >= best_reward and args.evaluate == 0:
                    torch.save(model.state_dict(),
                               args.dump_location+"train_easy_multigoal_fourier_d1_ae")

                    best_reward = np.mean(rewards_list)

                reach_list = []
                rewards_list = []
                accuracy_list = []
                episode_length_list = []
            reward_sum = 0
            episode_length = 0
            (image, depth, instruction), _, _= env.reset()
            # print(instruction)

            #-------------BUGG-------------
            episode_count += 1
            image_index = 0
            #------------------------------

            # depth =np.expand_dims(depth, axis=0)
            # image = np.concatenate((image, depth), axis=0)
            # Print instruction while evaluating and visualizing
            if args.evaluate != 0 and args.visualize == 1:
                print("Instruction: {} ".format(instruction))

            # Getting indices of the words in the instruction
            instruction_idx = []
            for word in instruction.split(" "):
                instruction_idx.append(env.word_to_idx[word])
            instruction_idx = np.array(instruction_idx)
            instruction_idx = torch.from_numpy(instruction_idx).view(1, -1)

        original_image = image
        image = torch.from_numpy(image).float()/255.0


