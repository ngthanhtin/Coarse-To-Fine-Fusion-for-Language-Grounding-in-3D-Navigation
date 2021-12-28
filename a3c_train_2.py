import torch.optim as optim
import env.env as grounding_env
import torchvision
from models.models import *
import math 
import matplotlib.pyplot as plt
import cv2
from ae.auto_encoder import Auto_Encoder_Model_PReLu
from models.aug_trajectory_buffer import aug_buffer
from utils.constants import WRONG_OBJECT_REWARD

log_file = 'train_easy_convolve_cf_new.log'
device = 'cpu'

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def train(rank, args, shared_model):
    
    torch.manual_seed(args.seed + rank)

    env = grounding_env.GroundingEnv(args)
    env.game_init()

    ae_model = None
    if args.auto_encoder:
        ae_model = Auto_Encoder_Model_PReLu()

    model = A3C_LSTM_GA(args, ae_model).to(device)
    
    if (args.load != "0"):
        print(str(rank) + " Loading model ... "+args.load)
        model.load_state_dict(
            torch.load(args.load, map_location=lambda storage, loc: storage))

    model.train()

    optimizer = optim.SGD(shared_model.parameters(), lr=args.lr)
    # optimizer = torch.optim.Adamax(filter(lambda p: p.requires_grad, shared_model.parameters()))
    # optimizer = torch.optim.Adam(shared_model.parameters(), lr=0.0001)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10000, T_mult=1, eta_min=0.00001, last_epoch=-1)
    if args.auto_encoder:
        ae_criterion = torch.nn.MSELoss()

    p_losses = []
    v_losses = []
    action_accuracy = []
    ae_losses = []

    (image, _, instruction), _, _ = env.reset()
    
    instruction_idx = []
    for word in instruction.split(" "):
        instruction_idx.append(env.word_to_idx[word])
    instruction_idx = np.array(instruction_idx).astype(float)

    original_image =  image
    image = torch.from_numpy(image).float()/255
    instruction_idx = torch.from_numpy(instruction_idx).view(1, -1).float()
    done = True

    episode_length = 0
    num_iters = 0
    entropy_coef =  0.01 #0.005
    decrease_factor = 0 #1e-6
    final_entropy_coeff = 0.01#0.005
    while True:
        # Sync with the shared model
        model.load_state_dict(shared_model.state_dict())
    
        if done:
            episode_length = 0
            cx = torch.Tensor(torch.zeros(1, 256)).to(device)
            hx = torch.Tensor(torch.zeros(1, 256)).to(device)

        else:
            cx = torch.Tensor(cx.data).to(device)
            hx = torch.Tensor(hx.data).to(device)

        values = []
        log_probs = []
        rewards = []
        entropies = []
        buffer = aug_buffer()
        buffer_list = []

        for step in range(args.num_steps):
            episode_length += 1
            tx = torch.Tensor(torch.from_numpy(np.array([episode_length])).float()).to(device) #.float()
            instruction_idx = instruction_idx.to(device).float()

            if args.auto_encoder:
                if original_image.shape == (3, 168, 300):
                    original_image = np.moveaxis(original_image, 0, 2)
    
                ae_input = original_image / 255.0
                
                ae_input = torch.Tensor(ae_input)
                ae_input = ae_input.permute(-1,0,1)
                ae_input = ae_input.unsqueeze(0)

                value, logit, (hx, cx), decoder = model((ae_input, torch.Tensor(image.unsqueeze(0)),
                                            torch.Tensor(instruction_idx),
                                            (tx, hx, cx)))

                decoder_image = decoder.squeeze().permute(1, 2, 0).detach().numpy()
                
                # plt.imshow(decoder_image)
                # plt.savefig('foo.png')
                # # plt.show()

                ae_loss = ae_criterion(decoder, ae_input)
            else:
                value, logit, (hx, cx) = model((torch.Tensor(image.unsqueeze(0)),
                                                torch.Tensor(instruction_idx),
                                                (tx, hx, cx)))

            

            prob = F.softmax(logit,dim=-1)
            log_prob = F.log_softmax(logit,dim=-1)
            entropy = -(log_prob * prob).sum(1)
            entropies.append(entropy)

            action = prob.multinomial(num_samples=1).data.float()
            log_prob = log_prob.gather(1, torch.Tensor(action).long())

            action = action.numpy()[0, 0]
            
            buffer_list.append([image, instruction_idx, (tx, hx, cx), action])

            (image, _, _), (reward, object_id), done = env.step(action)
            original_image =  image

            if done:
                if reward == WRONG_OBJECT_REWARD:
                    # sample proper instruction for this object
                    proper_instructions = sample_proper_instruction(object_id)
                    for proper_instruction in proper_instructions:
                        proper_instruction_idx = []
                        for word in proper_instruction.split(" "):
                            proper_instruction_idx.append(env.word_to_idx[word])
                        proper_instruction_idx = np.array(proper_instruction_idx).astype(float)
                        proper_instruction_idx = torch.from_numpy(proper_instruction_idx).view(1, -1).float()
                        for ob in buffer_list:
                            ob[1] = proper_instruction_idx
                            buffer.put(ob[0], ob[1], ob[2], ob[3])
                # delete buffer_list
                buffer_list.delete()

            done = done or episode_length >= args.max_episode_length

            if done:
                (image, _, instruction), _, _ = env.reset()

                instruction_idx = []
                for word in instruction.split(" "):
                    instruction_idx.append(env.word_to_idx[word])
                instruction_idx = np.array(instruction_idx)
                instruction_idx = torch.from_numpy(
                        instruction_idx).view(1, -1)

            original_image =  image
            image = torch.from_numpy(image).float()/255
            

            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                break

        R = torch.zeros(1, 1)
        if not done:
            tx = torch.Tensor(torch.from_numpy(np.array([episode_length])).float()).to(device)
            if args.auto_encoder:
                original_image = np.moveaxis(original_image, 0, 2)
                ae_input = original_image / 255.0
                
                ae_input = torch.Tensor(ae_input)
                ae_input = ae_input.permute(-1,0,1)
                ae_input = ae_input.unsqueeze(0)

                value, _, _, decoder = model((ae_input, torch.Tensor(image.unsqueeze(0)),
                                 torch.Tensor(instruction_idx), (tx, hx, cx)))
                ae_loss = ae_criterion(decoder, ae_input)
            else:
                value, _, _ = model((torch.Tensor(image.unsqueeze(0)),
                                    torch.Tensor(instruction_idx), (tx, hx, cx)))
            R = value.data

        values.append(torch.Tensor(R))
        policy_loss = 0
        value_loss = 0
        R = torch.Tensor(R)

        gae = torch.zeros(1, 1)

        if entropy_coef < final_entropy_coeff:
            entropy_coef = (1+decrease_factor)*entropy_coef
        else:
            entropy_coef = final_entropy_coeff
        
        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimataion
            delta_t = rewards[i] + args.gamma * \
                values[i + 1].data - values[i].data
            gae = gae * args.gamma * args.tau + delta_t

            policy_loss = policy_loss - \
                log_probs[i] * torch.Tensor(gae) - entropy_coef * entropies[i]

        # train for another goals
        crossentropy_loss, accuracy = training_action_prediction(buffer, model)
        #####

        optimizer.zero_grad()

        p_losses.append(policy_loss.data[0, 0])
        v_losses.append(value_loss.data[0, 0])
        if args.auto_encoder:
            ae_losses.append(ae_loss.item())

        if(len(p_losses) > 1000):
            num_iters += 1
            if args.auto_encoder:
                print(" ".join([
                    "Training thread: {}".format(rank),
                    "Num iters: {}K".format(num_iters),
                    "Avg policy loss: {}".format(np.mean(p_losses)),
                    "Avg value loss: {}".format(np.mean(v_losses)),
                    "Avg AE loss: {}".format(np.mean(ae_losses))]))
                
                with open(log_file, "a+") as f:
                    f.write(" ".join([
                        "Training thread: {}".format(rank),
                        "Num iters: {}K".format(num_iters),
                        "Avg policy loss: {}".format(np.mean(p_losses)),
                        "Avg value loss: {}".format(np.mean(v_losses)),
                        "Avg AE loss: {}\n".format(np.mean(ae_losses))]))
            else:
                print(" ".join([
                "Training thread: {}".format(rank),
                "Num iters: {}K".format(num_iters),
                "Avg policy loss: {}".format(np.mean(p_losses)),
                "Avg value loss: {}".format(np.mean(v_losses))]))
            
                with open(log_file, "a+") as f:
                    f.write(" ".join([
                        "Training thread: {}".format(rank),
                        "Num iters: {}K".format(num_iters),
                        "Avg policy loss: {}".format(np.mean(p_losses)),
                        "Avg value loss: {}\n".format(np.mean(v_losses))]))

            p_losses = []
            v_losses = []
            ae_losses = []

        if args.auto_encoder:
            if crossentropy_loss == -1 and accuracy == -1:
                (policy_loss + 0.5 * value_loss + ae_loss).backward() 
            else:
                (policy_loss + 0.5 * value_loss + ae_loss + 0.5*crossentropy_loss).backward() 
        else:
            if crossentropy_loss == -1 and accuracy == -1:
                (policy_loss + 0.5 * value_loss).backward() 
            else:
                (policy_loss + 0.5*value_loss + 0.5*crossentropy_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 40)  

        ensure_shared_grads(model, shared_model)
        optimizer.step()

def sample_proper_instruction(object_id):
    return 'huhu'

def training_action_prediction(buffer, model):
    if len(buffer) == 0:
        return -1, -1
    batch_size = 4
    batch = buffer.sample(batch_size)
    batch_state, batch_instr, label = np.array(batch.obs), np.array(batch.instr), np.array(batch.action)
    #
    episode_length = 0
    cx = torch.Tensor(torch.zeros(batch_size, 256)).to(device)
    hx = torch.Tensor(torch.zeros(batch_size, 256)).to(device)
    #
    batch_state = torch.from_numpy(batch_state).float()
    label = torch.tensor(label)

    with torch.cuda.device('cuda:0'):
        batch_state = Variable(torch.FloatTensor(batch_state)).cuda()
        label = label.cuda()

    value, logit, (hx, cx) = model()
    prob = F.softmax(logit,dim=-1)
    values, indices = logit.max(1)
    accuracy = torch.mean((indices.squeeze() == label).float())
    crossentropy_loss = F.cross_entropy(logit, label.long())

    return crossentropy_loss, accuracy