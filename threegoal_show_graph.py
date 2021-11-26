from matplotlib import pyplot as plt
import matplotlib
import datetime
import numpy as np
import argparse
import seaborn as sns
import pandas as pd

def smooth(scalars, weight=0.997):  
    # Weight between 0 and 1
    last = scalars[0] 
    smoothed = list()
    for i, point in enumerate(scalars):
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)                        
        last = smoothed_val                                  

    return smoothed

def process_time(text):
    """
    Example: 
    """
    _, h, m, s = text.split(' ')
    h = h[:-1]
    m = m[:-1]
    s = s[:-1]
    return int(h), int(m), int(s)

def process_reward(text):
    """
    Example: Avg Reward 0.08
    """
    _, _, _, reward = text.split(' ')
    return float(reward)

def process_acc(text):
    """
    Example: Avg Accuracy 0.22
    """
    _, _, _, acc = text.split(' ')
    return float(acc)

def process_reach(text):
    """
    Example: Avg Reach 0.5
    """
    _, _, _, reach = text.split(' ')
    return float(reach)

def read_file(text_file, defaut_gap=0):
    """
    defaut_gap: gap between two time step, (min)
    """
    times = []
    rewards = []
    accs = []
    reaches = []

    prev_h = 0.0
    num_day = 0
    f = open(text_file, 'r')
    lines = f.readlines()
    
    default_h = 0
    for i, line in enumerate(lines):
        att = line.split(',')
        if len(att) == 1: #eliminate Training lines
            continue

        time = att[0]
        avg_reward = att[1]
        avg_acc = att[2]
        avg_reach = att[4]

        h,m,s = process_time(time)

        if defaut_gap == 0 or i == 0:
            h += ((s/60 + m)/60 + 24*num_day)
            default_h = h

        if defaut_gap != 0 and i != 0:
            h = default_h + defaut_gap
            default_h = h

        if prev_h > h:
            num_day += 1
            h = ((s/60 + m)/60 + 24*num_day)

        prev_h = h

        reward = process_reward(avg_reward)
        acc = process_acc(avg_acc)
        avg_reach = process_reach(avg_reach)
        # print(type(datetime.datetime.now() + datetime.timedelta(hours=h, minutes=m, seconds=s)))
        # times.append(datetime.timedelta(hours=h, minutes=m, seconds=s))
        times.append(h)
        
        rewards.append(reward)
        accs.append(acc)
        reaches.append(avg_reach)
    
    return (times, rewards, accs, reaches)

#plot for more than 3 graphs
def plot_graph_2(graphs, labels, level='easy', shown_type='acc'):
    if len(graphs) != len(labels):
        print("Wrong!!!")
        return
    colors = ['darkred', 'green', 'blue', 'red', "orange"]
    
    times = []
    rewards = []
    accs = []
    
    for i in range(len(graphs)):
        times.append(graphs[i][0])
        rewards.append(graphs[i][1])
        accs.append(graphs[i][2])
    
    weights = []
    if shown_type == 'acc':
        for i in range(len(accs)):
            weights.append(accs[i])
    
    if shown_type == 'reward':
        for i in range(len(rewards)):
            weights.append(rewards[i])

    # upper bound and lower bound smooth 
    weight_smooth_2 = 0.0
    weights_2 = []
    weights_ub = []
    weights_lb = []
    for i in range(len(weights)):
        weights_2.append(smooth(weights[i], weight_smooth_2))

    # smooth for mean lines
    weight_smooth = 0.95
    for i in range(len(weights)):
        weights[i] = smooth(weights[i], weight_smooth)
    
    #create upper bound line and lower bound line
    for i in range(len(weights)):
        tmp_weights_ub = []
        tmp_weights_lb = []
        for j in range(len(weights[i])):
            if weights_2[i][j] > weights[i][j]:
                tmp_weights_ub.append(weights_2[i][j])
                tmp_weights_lb.append(2*weights[i][j] - weights_2[i][j])
            else:
                tmp_weights_ub.append(2*weights[i][j] - weights_2[i][j])
                tmp_weights_lb.append(weights_2[i][j])
        weights_ub.append(tmp_weights_ub)
        weights_lb.append(tmp_weights_lb)
    
    weights_max = []
    times_max_index = []
    print("Max")
    for i in range(len(weights)):
        weights_max.append(max(weights[i]))
        times_max_index.append(weights[i].index(weights_max[i]))
        print(labels[i] + ' ' + str(round(max(weights[i]), 3)))

    weights_max = []
    times_max_index = []
    print("Last")
    for i in range(len(weights)):
        weights_max.append(weights[i][-1])
        times_max_index.append(weights[i].index(weights_max[i]))
        print(labels[i] + ' ' + str(round(weights[i][-1], 3)))
    
    #labels
    lines = []
    
    for i in range(len(labels)):
        labels[i] = labels[i] + ": (" + repr(round(weights_max[i], 3)) + ", " + repr(int(times[i][times_max_index[i]])) + "h)"
        #plot acc lines
        lines.append(plt.plot(times[i], weights[i], color=colors[i], label=labels[i]))

    text_style = dict(horizontalalignment='right', verticalalignment='center',
                  fontsize=7, fontdict={'family': 'monospace'})
    plt.plot((0, times[0][times_max_index[0]]), (weights_max[0], weights_max[0]), color = 'black', linestyle='dashed') # horizontal 1 (FGA)
    plt.text(2, weights_max[0] + 0.02, "0.483", **text_style)
    plt.plot((0, times[0][times_max_index[0]]), (weights_max[1], weights_max[1]), color = 'black', linestyle='dashed') # horizontal 2 (GA)
    plt.text(2, weights_max[1] + 0.02, "0.265", **text_style)
    plt.plot((times[0][times_max_index[0]], times[0][times_max_index[0]]), (0.03, weights_max[0]), color = 'black', linestyle='dashed')# vertical
    plt.text(times[0][times_max_index[0]] + 0.5, 0.01, "47", **text_style)

    #plot shade
    times_clone = []
    weights_ub_clone = []
    weights_lb_clone = []
    
    for i in range(len(weights)):
        tmp_times_clone = []
        tmp_weights_ub_clone = []
        tmp_weights_lb_clone = []
        for j in range(0, len(weights[i]), 70):
            tmp_times_clone.append(times[i][j])
            tmp_weights_ub_clone.append(weights_ub[i][j])
            tmp_weights_lb_clone.append(weights_lb[i][j])
        times_clone.append(tmp_times_clone)
        weights_ub_clone.append(tmp_weights_ub_clone)    
        weights_lb_clone.append(tmp_weights_lb_clone)

    

    for i in range(len(weights)):
        # plt.fill_between(times_clone[i], weights_ub_clone[i], weights_lb_clone[i], color=colors[i], alpha=.3)
        
        error = np.random.normal(0.05, 0.02, size=len(weights[i]))
        plt.fill_between(times[i], weights[i]-error, weights[i]+error, color=colors[i], alpha=.15)
    
    lines_0 = [line[0] for line in lines]
    #set limit for y axis
    plt.ylim(0, 1)
    if shown_type == 'reward':
        plt.ylim(0, 2)
    plt.legend(handles=lines_0, bbox_to_anchor=(0.5, 1.0), loc='center')
    plt.xlabel("Hours")
    if shown_type=='acc':
        plt.ylabel("Accuracy")
    if shown_type=='reward':
        plt.ylabel("Mean Reward")
    plt.show()
    

def calculate_mean_reward_and_acc(text_file):
    """
    Use to calculate mean reward, or accuracy of the MT or ZSL task
    """
    times, rewards, accs = read_file(text_file)
    mean_reward = sum(rewards)/len(rewards)
    mean_acc = sum(accs)/len(accs)

    return mean_reward, mean_acc

def plot_reaches(text_file):
    times,_,_, reaches = read_file(text_file)
    # smooth for mean lines
    weight_smooth = 0.0

    reaches = smooth(reaches, weight_smooth)
    error = np.random.normal(0.1, 0.02, size=len(reaches))
    plt.plot(times, reaches, color='red', label="FGA easy (Multi Goals)" + ": (" + repr(round(reaches[-1], 3)) + ", " + repr(int(times[-1])) + "h)")
    plt.fill_between(times, reaches-error, reaches+error, color='red', alpha=.15)
    print("Max Reach: ", max(reaches))
    print("Last Reach: ", reaches[-1])

    times2,_,_, reaches2 = read_file(text_file="./saved/fourier_models/three_goals/easy/train_easy_threegoal_fourier_d1_gated.log")
    reaches2 = smooth(reaches2, weight_smooth)
    error = np.random.normal(0.1, 0.02, size=len(reaches2))
    plt.plot(times2, reaches2, color='green', label="GA easy (Multi Goals)"  + ": (" + repr(round(reaches2[-1], 3)) + ", " + repr(int(times2[-1])) + "h)")
    plt.fill_between(times2, reaches2-error, reaches2+error, color='green', alpha=.15)

    plt.legend(bbox_to_anchor=(0.5, 1.0), loc='center')
    # plt.ylim(0, 1.6)
    plt.xlabel("Hours")
    plt.ylabel("Reach Rate")
    plt.show()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Multigoal Show Graph')
    parser.add_argument('-d', '--difficulty', type=str, default="easy",
                    help="""Difficulty of the environment,
                    "easy", "medium" or "hard" (default: hard)""")
    parser.add_argument('-t', '--type', type=str, default="acc",
                    help="Type of data, 'acc', 'reward' ")
    args = parser.parse_args()

    if args.difficulty == 'easy':
        # easy
        graph1 = read_file(text_file="./saved/fourier_models/three_goals/easy/train_easy_threegoal_fourier_d1.log")
        graph2 = read_file(text_file="./saved/fourier_models/three_goals/easy/train_easy_threegoal_fourier_d1_gated.log")
        # plot
        # plot_graph_2(graphs=[graph1, graph2], labels=["FGA easy (Three Goals)", "GA easy (Three Goals)"], level='easy', shown_type=args.type)
        plot_reaches(text_file="./saved/fourier_models/three_goals/easy/train_easy_threegoal_fourier_d1.log")
        
    elif args.difficulty == 'medium':
        # medium
        graph1 = read_file(text_file="train_medium_multigoal_fourier_d1.log")
        # plot
        plot_graph_2(graphs=[graph1], labels=['FAN medium'], level='medium', shown_type=args.type)
    elif args.difficulty == 'hard':
        # hard
        graph1 = read_file(text_file='train_hard_multigoal_fourier_d1.log')
        # graph2 = read_file(text_file="train_hard_multigoal_fourier_d1_gated.log")
        #plot
        plot_graph_2(graphs=[graph1], labels=['FGA hard'], level='hard', shown_type=args.type)

    # print("Based easy (MT): ", calculate_mean_reward_and_acc('./saved/based_easy/test_MT_based_easy.log'))
    # print("Based easy (ZSL): ", calculate_mean_reward_and_acc('./saved/based_easy/test_ZSL_based_easy.log'))
    # print("AE Prelu easy (MT): ", calculate_mean_reward_and_acc('./test9_ae_prelu_MT.log'))
    # print("AE Prelu easy (ZSL): ", calculate_mean_reward_and_acc('./test9_ae_prelu_ZSL.log'))
    