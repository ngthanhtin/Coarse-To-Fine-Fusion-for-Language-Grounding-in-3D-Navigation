import numpy as np
import torch
import torch.nn.functional as F
import time
import cv2
import env.env as grounding_env
from models.models import A3C_LSTM_GA
from ae.auto_encoder import Auto_Encoder_Model_PReLu
from utils.constants import *

device='cpu'
log_file = 'train_convolve_failed.log'

def test(rank, args, shared_model):

    torch.manual_seed(args.seed + rank)

    env = grounding_env.GroundingEnv(args)
    env.game_init()

    ae_model = None
    if args.auto_encoder:
        ae_model = Auto_Encoder_Model_PReLu()

    model = A3C_LSTM_GA(args, ae_model).to(device)
    if (args.load != "0"):
        print("Loading model ... "+args.load)
        model.load_state_dict(
            torch.load(args.load, map_location=lambda storage, loc: storage))
            
    model.eval()

    (image, depth, instruction), _, _ = env.reset()
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
    rewards_list = []
    accuracy_list = []
    episode_length_list = []
    num_episode = 0
    best_reward = 0.0
    test_freq = 50
    while True:
        episode_length += 1
        if done:
            if (args.evaluate == 0):
                model.load_state_dict(shared_model.state_dict())

            cx = torch.Tensor(torch.zeros(1, 256)).to(device)
            hx = torch.Tensor(torch.zeros(1, 256)).to(device)
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

            value, logit, (hx, cx), decoder = model(
                (ae_input, torch.Tensor(image.unsqueeze(0)),
                 torch.Tensor(instruction_idx), (tx, hx, cx)))
        else:
            original_image = np.moveaxis(original_image, 0, 2)
            # cv2.imwrite('foo.png', cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))
            value, logit, (hx, cx) = model(
                    (torch.Tensor(image.unsqueeze(0)),
                    torch.Tensor(instruction_idx), (tx, hx, cx)))

        prob = F.softmax(logit, dim=-1)
        action = prob.max(1)[1].data.numpy()

        (image, depth, _), reward, done = env.step(action[0])

        # depth =np.expand_dims(depth, axis=0)
        # image = np.concatenate((image, depth), axis=0)

        done = done or episode_length >= args.max_episode_length
        reward_sum += reward    

        if done:
            # print(reward)
            num_episode += 1
            rewards_list.append(reward_sum)
            # Print reward while evaluating and visualizing
            if args.evaluate != 0 and args.visualize == 1:
                print("Total reward: {}".format(reward_sum))

            episode_length_list.append(episode_length)
            if reward == CORRECT_OBJECT_REWARD:
                accuracy = 1
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
                    "Best Reward {}".format(best_reward)]))
                with open(log_file, "a+") as f:
                    f.write(" ".join([
                        "Time {},".format(time.strftime("%Hh %Mm %Ss",
                                        time.gmtime(time.time() - start_time))),
                        "Avg Reward {},".format(np.mean(rewards_list)),
                        "Avg Accuracy {},".format(np.mean(accuracy_list)),
                        "Avg Ep length {},".format(np.mean(episode_length_list)),
                        "Best Reward {}\n".format(best_reward)]))
                if np.mean(rewards_list) >= best_reward and args.evaluate == 0:
                    torch.save(model.state_dict(),
                               args.dump_location+"train_convolve_failed")

                    best_reward = np.mean(rewards_list)

                rewards_list = []
                accuracy_list = []
                episode_length_list = []
            reward_sum = 0
            episode_length = 0
            (image, depth, instruction), _, _ = env.reset()
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
            instruction_idx = np.array(instruction_idx)
            instruction_idx = torch.from_numpy(instruction_idx).view(1, -1)

        original_image = image
        image = torch.from_numpy(image).float()/255.0


