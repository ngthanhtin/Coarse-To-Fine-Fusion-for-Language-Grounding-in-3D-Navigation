import vizdoom
import argparse
import env as grounding_env
import numpy as np
import cv2
from auto_encoder import Auto_Encoder_Model, Auto_Encoder_Model_Original
import torch
parser = argparse.ArgumentParser(description='Grounding Environment Test')
parser.add_argument('-l', '--max-episode-length', type=int, default=30,
                    help='maximum length of an episode (default: 30)')
parser.add_argument('-d', '--difficulty', type=str, default="hard",
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
                    help="""Visualize the envrionment (default: 1,
                    change to 0 for faster training)""")
parser.add_argument('--sleep', type=float, default=0,
                    help="""Sleep between frames for better
                    visualization (default: 0)""")
parser.add_argument('-t', '--use_train_instructions', type=int, default=1,
                    help="""0: Use test instructions, 1: Use train instructions
                    (default: 1)""")
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

if __name__ == '__main__':
    args = parser.parse_args()
    env = grounding_env.GroundingEnv(args)
    env.game_init()

    num_episodes = 0
    rewards_per_episode = []
    reward_sum = 0
    is_final = 1

    save_path = '/home/tinvn/TIN/NLP_RL_Code/data/depth_medium/'
    image_index = 0

    device = 'cuda:0'
    # model = Auto_Encoder_Model_Original()#.to(device)
    # weight_path = './ae/ae_full_original.pth'
    # print('load initial weights CDAE from: %s'%(weight_path))
    # model.load_state_dict(torch.load(weight_path))

    while num_episodes < 400:
        if is_final:
            (image, depth, instruction), _, _, _ = env.reset()
            print("Instruction: {}".format(instruction))

        # Take a random action
        (image, depth, instruction), reward, is_final, _ = \
            env.step(np.random.randint(3))
        
        # reconstruct image
        # image = np.moveaxis(image, 0, 2)
        # ae_input = image / 255.0
        # ae_input = torch.Tensor(ae_input)
        # ae_input = ae_input.permute(-1,0,1)
        # ae_input = ae_input.unsqueeze(0)

        # encoder = model.forward_pass(ae_input)
        # decoder = model.reconstruct_pass(encoder)
        # decoder_image = decoder.squeeze().permute(1, 2, 0).detach().numpy()
        # cv2.imshow("assaa", decoder_image)
        # cv2.waitKey(0)
        # save image
        # saved_image = np.moveaxis(depth, 0, -1) # for rgb image
        saved_image = np.expand_dims(depth, axis=2)
        
        # saved_image = cv2.cvtColor(saved_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(save_path+'{}_depth_medium.png'.format(image_index), saved_image)
        image_index += 1
        if image_index == 5000:
            break
        reward_sum += reward

        if is_final:
            print("Total Reward: {}".format(reward_sum))
            rewards_per_episode.append(reward_sum)
            num_episodes += 1
            reward_sum = 0
            if num_episodes % 10 == 0:
                print("Avg Reward per Episode: {}".format(
                    np.mean(rewards_per_episode)))
