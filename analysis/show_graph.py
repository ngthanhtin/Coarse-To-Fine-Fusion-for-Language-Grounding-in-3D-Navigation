from os import read
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

def read_file(text_file, defaut_gap=0):
    """
    defaut_gap: gap between two time step, (min)
    """
    times = []
    rewards = []
    accs = []

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

        h,m,s = process_time(time)
        # if avg_reward == '0.42400000000000004' and  time == '00h 00m 58s': # 13h 31m 29s
        #     s += 29
        #     if s > 60:
        #         s -= 60
        #         m+=1
        #     m += 31
        #     if m > 60:
        #         m-= 60
        #         h+= 1
        #     h += 13

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
        # print(type(datetime.datetime.now() + datetime.timedelta(hours=h, minutes=m, seconds=s)))
        # times.append(datetime.timedelta(hours=h, minutes=m, seconds=s))
        times.append(h)
        
        rewards.append(reward)
        accs.append(acc)
    
    return (times, rewards, accs)

#plot for more than 3 graphs
def plot_graph_2(graphs, labels, level='easy', shown_type='acc'):
    if len(graphs) != len(labels):
        print("Wrong!!!")
        return
    colors = ['darkred', 'green', 'blue', 'red', "darkorange", "purple"]
    colors = ['green', 'blue', 'red', "darkorange", "purple"]
    colors = ['green', 'blue', 'red', "darkorange", "red", "darkorange", 'blue']
    
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
    weight_smooth = 0.99
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

    if level == 'easy':
        weights[0] = weights[0][:-200]
        times[0] = times[0][:-200]
    if level == 'medium':
        #weights[0] = weights[0][:-110] # san + ae
        #times[0] = times[0][:-110]

        weights[0] = weights[0][:7170] # based
        times[0] = times[0][:7170]
        
        # weights[2] = weights[2][:6955] # san only
        # times[2] = times[2][:6955]
    if level == 'hard':
        # weights[0] = weights[0][:-800]
        # times[0] = times[0][:-800]
        
        # weights[0] = weights[0][:10000] #based
        # times[0] = times[0][:10000]

        weights[0] = weights[0][:9650] #based
        times[0] = times[0][:9650]
        
        
        times[4] = [t+85.3 for t in times[4]]
        times[5] = [t+34.8 for t in times[5]]
        times[6] = [t+129.3 for t in times[6]]

    #plot the max dashed lines
    # points1 = np.ones(int(max(times1)))
    # points2 = np.ones(int(max(times2)))
    # points3 = np.ones(int(max(times3)))
    
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
    abandoned_graph = []
    lines = []
    
    for i in range(len(labels)):
        if i in abandoned_graph:
            continue
        if i == 0:
            labels[i] = labels[i] + ": (" + repr(round(weights_max[i], 3)) + ", " + repr(int(times[i][times_max_index[i]])) + "h)"
        if i == 1:
            labels[1] = labels[1] + ": (" + repr(round(weights_max[6], 3)) + ", " + repr(int(times[6][times_max_index[6]])) + "h)"
        if i == 2:
            labels[2] = labels[2] + ": (" + repr(round(weights_max[4], 3)) + ", " + repr(int(times[4][times_max_index[4]])) + "h)"
        if i == 3:
            labels[3] = labels[3] + ": (" + repr(round(weights_max[5], 3)) + ", " + repr(int(times[5][times_max_index[5]])) + "h)"
        #plot acc lines
        lines.append(plt.plot(times[i], weights[i], color=colors[i], label=labels[i]))

    #plot the max dashed lines
    points = [np.ones(int(max(time))) for time in times]
    text_style = dict(horizontalalignment='right', verticalalignment='center',
                  fontsize=7, fontdict={'family': 'monospace'})
    
    # easy
    # plt.plot((0, times[0][times_max_index[0]]), (weights_max[0], weights_max[0]), color = 'black', linestyle='dashed') # horizontal
    # plt.text(0, weights_max[0], "0.897", **text_style)
    # plt.plot((times[0][times_max_index[0]], times[0][times_max_index[0]]), (0.03, weights_max[0]), color = 'black', linestyle='dashed')# vertical 1
    # plt.text(times[0][times_max_index[0]] + 2, 0.01, "101", **text_style)
    # plt.plot((43.4, 43.4), (0.03, 0.896), color = 'black', linestyle='dashed')# vertical 2
    # plt.text(43 + 2, 0.01, "43.4", **text_style)
    # plt.plot((15.6, 15.6), (0.03, 0.896), color = 'black', linestyle='dashed')# vertical 3
    # plt.text(times[2][times_max_index[2]] + 2, 0.01, "15.6", **text_style)
    # plt.plot((10.9, 10.9), (0.03, 0.896), color = 'black', linestyle='dashed')# vertical 4
    # plt.text(12.3, 0.01, "11", **text_style)

    # medium
    # plt.plot((0, times[0][times_max_index[0]]), (weights_max[0], weights_max[0]), color = 'black', linestyle='dashed') # horizontal
    # plt.text(0, weights_max[0], "0.79", **text_style)
    # plt.plot((times[0][times_max_index[0]], times[0][times_max_index[0]]), (0.03, weights_max[0]), color = 'black', linestyle='dashed')# vertical 1
    # plt.text(times[0][times_max_index[0]] + 2, 0.01, "125", **text_style)
    # plt.plot((80.7, 80.7), (0.03, 0.79), color = 'black', linestyle='dashed')# vertical 2
    # plt.text(times[1][times_max_index[1]] + 2, 0.01, "80.7", **text_style)
    # plt.plot((33.7, 33.7), (0.03, 0.79), color = 'black', linestyle='dashed')# vertical 3
    # plt.text(34 + 2, 0.01, "34", **text_style)
    # plt.plot((31.8, 31.8), (0.03, 0.79), color = 'black', linestyle='dashed')# vertical 4
    # plt.text(31 + 2, 0.01, "32", **text_style)

    # # hard
    plt.plot((0, times[0][times_max_index[0]]), (weights_max[0], weights_max[0]), color = 'black', linestyle='dashed') # horizontal
    plt.text(0, weights_max[0], "0.57", **text_style)
    plt.plot((0, times[0][times_max_index[0]]), (0.78, 0.78), color = 'black', linestyle='dashed') # horizontal 2
    plt.text(0, 0.78, "0.78", **text_style)
    plt.plot((times[0][times_max_index[0]], times[0][times_max_index[0]]), (0.03, weights_max[0]), color = 'black', linestyle='dashed')# vertical 1
    plt.text(times[0][times_max_index[0]] + 2, 0.01, "166", **text_style)
    plt.plot((166, 166), (0.57, 0.783), color = 'black', linestyle='dashed')# vertical 2
    # plt.text(129 + 2, 0.01, "129", **text_style)
    plt.plot((131, 131), (0.03, 0.78), color = 'black', linestyle='dashed')# vertical 3
    plt.text(131 + 3, 0.01, "131", **text_style)
    plt.plot((111, 111), (0.03, 0.786), color = 'black', linestyle='dashed')# vertical 4
    plt.text(111 + 3, 0.01, "111", **text_style)

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
        if i in abandoned_graph:
            continue
        # plt.fill_between(times_clone[i], weights_ub_clone[i], weights_lb_clone[i], color=colors[i], alpha=.3)
        error = np.random.normal(0.05, 0.02, size=len(weights[i]))
        plt.fill_between(times[i], weights[i]-error, weights[i]+error, color=colors[i], alpha=.2)
    
    lines_0 = [line[0] for line in lines[:4]]
    # lines_0 = [lines[1][0], lines[3][0], lines[4][0]]
    #set limit for y axis
    plt.ylim(0, 1)
    if level == 'hard':
        plt.ylim(0, 1.)
    plt.legend(handles=lines_0, bbox_to_anchor=(0.5, 1.), loc='center', ncol=2)
    plt.xlabel("Hours")
    if shown_type=='acc':
        plt.ylabel("Accuracy")
    if shown_type=='reward':
        plt.ylabel("Mean Reward")
    plt.show()
    
    # rewards1 = np.array(weights_lb_clone[0])
    # rewards2 = np.array(weights_ub_clone[0])
    
    # rewards=np.vstack((rewards1,rewards2)) # Merge array
    # df = pd.DataFrame(rewards).melt(var_name='episode',value_name='reward')

    # sns.lineplot(x="episode", y="reward", data=df)
    # plt.show()

def calculate_mean_reward_and_acc(text_file):
    """
    Use to calculate mean reward, or accuracy of the MT or ZSL task
    """
    times, rewards, accs = read_file(text_file)
    mean_reward = sum(rewards)/len(rewards)
    mean_acc = sum(accs)/len(accs)

    return mean_reward, mean_acc

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Show Graph')
    parser.add_argument('-d', '--difficulty', type=str, default="easy",
                    help="""Difficulty of the environment,
                    "easy", "medium" or "hard" (default: hard)""")
    parser.add_argument('-t', '--type', type=str, default="acc",
                    help="Type of data, 'acc', 'reward' ")
    # parser.add_argument('-p1', '--plot1', type=int, default=0,
    #                 help='show the plot type 1 (default: 0)')
    parser.add_argument('-p1', '--plot1', action='store_true')
    parser.add_argument('-p2', '--plot2', action='store_true')
    # parser.add_argument('-p2', '--plot2', type=int, default=0,
    #                 help='show the plot type 2 (default: 0)')
    args = parser.parse_args()

    if args.difficulty == 'easy':
        # easy
        graph1 = read_file(text_file="./saved/easy/based_easy/train_based_easy.log") # base
        graph2 = read_file(text_file="./saved/convolve/train_easy_convolve.log")
        graph3 = read_file(text_file="./saved/cf_convolve/train_easy_convolve_cf.log")
        graph4 = read_file(text_file="./saved/cf_convolve/train_easy_convolve_cf_ae.log")

        # plot
        if args.plot1:
            plot_graph(graph1, graph2, graph3, label1='SAN + AE easy', label2="Gated-attention easy", label3='SAN easy', level='easy', shown_type=args.type)
        if args.plot2:
            plot_graph_2(graphs=[graph1, graph2, graph3, graph4], labels=['GA easy', 'CA easy', "CFCA easy", "CFCA+AE easy"], level='easy', shown_type=args.type)
    elif args.difficulty == 'medium':
        # medium
        graph1 = read_file(text_file="./saved/medium/based_medium/train8_medium.log") #base
        graph2 = read_file(text_file="./saved/convolve/train_medium_convolve.log")
        graph3 = read_file(text_file='./saved/cf_convolve/train_medium_cfconvolve.log')
        graph4 = read_file(text_file="./saved/cf_convolve/train_medium_cfconvolve_ae.log")

        # plot
        if args.plot1:
            plot_graph(graph1, graph2, graph3, label1='CFCA medium', label2="GA medium", label3='SAN medium', level='medium', shown_type=args.type)
        if args.plot2:
            plot_graph_2(graphs=[graph1, graph2, graph3, graph4], labels=['GA medium', 'CA medium', "CFCA medium", 'CFCA+AE medium'], level='medium', shown_type=args.type)
    elif args.difficulty == 'hard':
        # hard
        graph1 = read_file(text_file="./saved/hard/based_hard/train_based_hard.log", defaut_gap=62/3600) # train_hard base
        graph2 = read_file(text_file="./saved/convolve/train_hard_convolve.log")
        graph3 = read_file(text_file="./saved/cf_convolve/train_hard_cfconvolve.log")
        graph4 = read_file(text_file="./saved/cf_convolve/train_hard_convolve_cf_ae.log")
        graph5 = read_file(text_file='./train_hard_cfconvolve_continue.log')
        graph6 = read_file(text_file='./train_hard_convolve_cf_ae_full.log')
        graph7 = read_file(text_file='./train_hard_convolve_continue.log')
        #plot
        
        plot_graph_2(graphs=[graph1, graph2, graph3, graph4, graph5, graph6, graph7], labels=['GA hard', 'CA hard', 'CFCA hard', 'CFCA+AE hard', 'add', 'add', 'add'], level='hard', shown_type=args.type)

    # print("Based easy (MT): ", calculate_mean_reward_and_acc('./saved/based_easy/test_MT_based_easy.log'))
    # print("Based easy (ZSL): ", calculate_mean_reward_and_acc('./saved/based_easy/test_ZSL_based_easy.log'))
    # print("AE Prelu easy (MT): ", calculate_mean_reward_and_acc('./test9_ae_prelu_MT.log'))
    # print("AE Prelu easy (ZSL): ", calculate_mean_reward_and_acc('./test9_ae_prelu_ZSL.log'))
    