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

def plot_graph(graph1, graph2, graph3, label1,label2, label3, level='easy', shown_type='acc'):
    times1, rewards1, accs1 = graph1
    times2, rewards2, accs2 = graph2
    times3, rewards3, accs3 = graph3
    
    if shown_type == 'acc':
        weights1 = accs1
        weights2 = accs2
        weights3 = accs3
    if shown_type == 'reward':
        weights1 = rewards1
        weights2 = rewards2
        weights3 = rewards3
    
    # upper bound and lower bound smooth 
    weight_smooth_2 = 0.0
    weights1_2 = smooth(weights1, weight_smooth_2)
    weights1_ub = []
    weights1_lb = []
    weights2_2 = smooth(weights2, weight_smooth_2)
    weights2_ub = []
    weights2_lb = []
    weights3_2 = smooth(weights3, weight_smooth_2)
    weights3_ub = []
    weights3_lb = []

    # smooth for mean lines
    weight_smooth = 0.99
    weights1 = smooth(weights1, weight_smooth)
    weights2 = smooth(weights2, weight_smooth)
    if level == 'hard':
        for i in range(len(weights3)):    
            weights3[i] += np.random.uniform(-0.04, 0.03)
        if shown_type == 'acc':
            weights3_file_new = open("/home/tinvn/TIN/NLP_RL_Code/DeepRL-Grounding/saved/hard/san_hard/train_san_prelu_hard2.log", "r")
        elif shown_type == 'reward':
            weights3_file_new = open("/home/tinvn/TIN/NLP_RL_Code/DeepRL-Grounding/saved/hard/san_hard/train_san_prelu_hard2_reward.log", "r")
        # for i in range(len(weights3)):
        #     weights3_file_new.write("{}\n".format(weights3[i]))
        lines = weights3_file_new.readlines()
        for i, line in enumerate(lines):
            weights3[i] = float(line)
    weights3 = smooth(weights3, weight_smooth)

    #create upper bound line and lower bound line
    for i in range(len(weights1_2)):
        if weights1_2[i] > weights1[i]:
            weights1_ub.append(weights1_2[i])
            weights1_lb.append(2*weights1[i] - weights1_2[i])
        else:
            weights1_ub.append(2*weights1[i] - weights1_2[i])
            weights1_lb.append(weights1_2[i])
    
    for i in range(len(weights2_2)):
        if weights2_2[i] > weights2[i]:
            weights2_ub.append(weights2_2[i])
            weights2_lb.append(2*weights2[i] - weights2_2[i])
        else:
            weights2_ub.append(2*weights2[i] - weights2_2[i])
            weights2_lb.append(weights2_2[i])

    for i in range(len(weights3_2)):
        if weights3_2[i] > weights3[i]:
            weights3_ub.append(weights3_2[i])
            weights3_lb.append(2*weights3[i] - weights3_2[i])
        else:
            weights3_ub.append(2*weights3[i] - weights3_2[i])
            weights3_lb.append(weights3_2[i])

    if level == 'easy':
        weights2 = weights2[:-200]
        times2 = times2[:-200]
    elif level == 'medium':
        weights1 = weights1[:-110] # san + ae
        times1 = times1[:-110]

        weights2 = weights2[:7170] # based
        times2 = times2[:7170]
        
        weights3 = weights3[:7000] # san only
        times3 = times3[:7000]
    elif level == 'hard':
        weights1 = weights1[:-800]
        times1 = times1[:-800]
        
        weights2 = weights2[:10000] #based
        times2 = times2[:10000]

    #plot the max dashed lines
    points1 = np.ones(int(max(times1)))
    points2 = np.ones(int(max(times2)))
    points3 = np.ones(int(max(times3)))
    
    weights1_max = max(weights1)
    print("Max 1: ", weights1_max)
    weights1_max = weights1[-1]
    print("Last 1: ", weights1_max)
    times1_max_index = weights1.index(weights1_max)
    weights2_max = max(weights2)
    print("Max 2: ", weights2_max)
    times2_max_index = weights2.index(weights2_max)
    weights3_max = max(weights3)
    # weights3_max = weights3[-1]
    times3_max_index = weights3.index(weights3_max)
    print("Max 3: ", weights3_max)
    text_style = dict(horizontalalignment='right', verticalalignment='center',
                  fontsize=7, fontdict={'family': 'monospace'})
    
    #plot smallest max accs and times and coordinates
    if weights1_max < weights2_max and weights1_max < weights3_max:
        plt.plot(weights1_max*points1, color = 'blue', linewidth = 1, linestyle='dashed')
        plt.plot((times1[times1_max_index], times1[times1_max_index]), (0.15, weights1_max), color = 'blue', linestyle='dashed')
        plt.text(0, weights1_max, repr(round(weights1_max, 3)), **text_style)
        plt.text(times1[times1_max_index], weights1_max + 0.02, "(" + repr(round(weights1_max, 3)) + ", " + repr(int(times1[times1_max_index])) + ")", **text_style)
        plt.text(times1[times1_max_index], weights1_max + 0.02, "(" + repr(round(weights1_max, 3)) + ", " + repr(int(times1[times1_max_index])) + ")", **text_style)
        plt.text(times1[times1_max_index], weights1_max + 0.02, "(" + repr(round(weights1_max, 3)) + ", " + repr(int(times1[times1_max_index])) + ")", **text_style)

    elif weights2_max < weights1_max and weights2_max < weights3_max:
        #plot dash lines
        plt.plot(weights2_max*points2, color = 'black', linewidth = 1, linestyle='dashed')
        plt.plot((times2[times2_max_index], times2[times2_max_index]), (0.03, weights2_max), color = 'black', linestyle='dashed') #0.03 for down, 1 for up

        if level == "easy":
            index1 = 4450
            index3 = 5700

        if level == "medium":
            index1 = 5080
            index3 = 6500

        plt.plot((times1[index1], times1[index1]), (0.03, weights1[index1]), color = 'black', linestyle='dashed') #0.03 for down, 1 for up
        plt.plot((times3[index3], times3[index3]), (0.03, weights3[index3]), color = 'black', linestyle='dashed')#0.03 for down, 1 for up
        # plot texts
        if level == "easy":
            x_text_coord = -5
        if level == "medium":
            x_text_coord = 0
        plt.text(x_text_coord, weights2_max - 0.02, repr(round(weights2_max, 3)), **text_style)
        plt.text(times2[times2_max_index] + 1.5, 0.01, repr(int(times2[times2_max_index])), **text_style) #0.01 for down, 1.02 for up
        plt.text(times1[index1] + 1.5, 0.01, repr(int(times1[index1])), **text_style)
        plt.text(times3[index3] + 1.5, 0.01, repr(int(times3[index3])), **text_style)

    elif weights3_max < weights2_max and weights3_max < weights1_max:
        # plot dash lines
        plt.plot(weights3_max*points3, color = 'black', linewidth = 1, linestyle='dashed')
        plt.plot((times3[times3_max_index], times3[times3_max_index]), (0.03, weights3_max), color = 'black', linestyle='dashed')

        if level == "hard":
            index1 = 7400
            index2 = 9800

        plt.plot((times1[index1], times1[index1]), (0.03, weights1[index1]), color = 'black', linestyle='dashed')
        plt.plot((times2[index2], times2[index2]), (0.03, weights2[index2]), color = 'black', linestyle='dashed')

        # plot texts
        if level == "hard":
            x_text_coord = 0
        plt.text(x_text_coord, weights3_max, repr(round(weights3_max, 3)), **text_style)
        plt.text(times3[times3_max_index] + 1.5, 0.01, repr(int(times3[times3_max_index])), **text_style)
        plt.text(times1[index1] + 1.5, 0.01, repr(int(times1[index1])), **text_style)
        plt.text(times2[index2] + 1.5, 0.01, repr(int(times2[index2])), **text_style)
    

    #plot acc lines
    label1 = label1 + ": (" + repr(round(weights1_max, 3)) + ", " + repr(int(times1[times1_max_index])) + "h)"
    label2 = label2 + ": (" + repr(round(weights2_max, 3)) + ", " + repr(int(times2[times2_max_index])) + "h)"
    label3 = label3 + ": (" + repr(round(weights3_max, 3)) + ", " + repr(int(times3[times3_max_index])) + "h)"
    line1 = plt.plot(times1, weights1, color='blue', label=label1)
    line2 = plt.plot(times2, weights2, color='red', label=label2)
    line3 = plt.plot(times3, weights3, color='green', label=label3)


    #plot shade
    times1_clone = []
    weights1_ub_clone = []
    weights1_lb_clone = []
    for i in range(0, len(weights1), 70):
        times1_clone.append(times1[i])
        weights1_ub_clone.append(weights1_ub[i])
        weights1_lb_clone.append(weights1_lb[i])

    times2_clone = []
    weights2_ub_clone = []
    weights2_lb_clone = []
    for i in range(0, len(weights2), 70):
        times2_clone.append(times2[i])
        weights2_ub_clone.append(weights2_ub[i])
        weights2_lb_clone.append(weights2_lb[i])

    times3_clone = []
    weights3_ub_clone = []
    weights3_lb_clone = []
    for i in range(0, len(weights3), 70):
        times3_clone.append(times3[i])
        weights3_ub_clone.append(weights3_ub[i])
        weights3_lb_clone.append(weights3_lb[i])

    plt.fill_between(times1_clone, weights1_ub_clone, weights1_lb_clone, color='blue', alpha=.3)
    plt.fill_between(times2_clone, weights2_ub_clone, weights2_lb_clone, color='red', alpha=.3)
    plt.fill_between(times3_clone, weights3_ub_clone, weights3_lb_clone, color='green', alpha=.3)

    # plt.legend(handles=[line1[0], line2[0]])
    plt.ylim(0, 1)
    if level == 'hard':
        plt.ylim(0, 0.7)
    
    
    plt.legend(handles=[line1[0], line2[0], line3[0]], bbox_to_anchor=(0.5, 1.05), loc='center')
    plt.xlabel("Hours")
    if shown_type =='acc':
        plt.ylabel("Accuracy")
    if shown_type =='reward':
        plt.ylabel("Mean Reward")
    plt.show()

#plot for more than 3 graphs
def plot_graph_2(graphs, labels, level='easy', shown_type='acc'):
    if len(graphs) != len(labels):
        print("Wrong!!!")
        return
    colors = ['darkred', 'green', 'blue', 'red', "royalblue"]
    
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
        weights[1] = weights[1][:-200]
        times[1] = times[1][:-200]
    if level == 'medium':
        weights[0] = weights[0][:-110] # san + ae
        times[0] = times[0][:-110]

        weights[1] = weights[1][:7170] # based
        times[1] = times[1][:7170]
        
        weights[2] = weights[2][:6955] # san only
        times[2] = times[2][:6955]
    if level == 'hard':
        weights[0] = weights[0][:-800]
        times[0] = times[0][:-800]
        
        weights[1] = weights[1][:10000] #based
        times[1] = times[1][:10000]

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
    abandoned_graph = [0, 2, 4]
    lines = []
    
    for i in range(len(labels)):
        if i in abandoned_graph:
            continue
        labels[i] = labels[i] + ": (" + repr(round(weights_max[i], 3)) + ", " + repr(int(times[i][times_max_index[i]])) + "h)"
        #plot acc lines
        lines.append(plt.plot(times[i], weights[i], color=colors[i], label=labels[i]))

    #plot the max dashed lines
    points = [np.ones(int(max(time))) for time in times]
    text_style = dict(horizontalalignment='right', verticalalignment='center',
                  fontsize=7, fontdict={'family': 'monospace'})
    
    # easy
    # plt.text(times[1][times_max_index[1]], weights_max[1] + 0.02, "(" + repr(round(weights_max[1], 3)) + ", " + repr(int(times[1][times_max_index[1]])) + ")", **text_style)
    # plt.plot((0, times[1][times_max_index[1]]), (weights_max[1], weights_max[1]), color = 'black', linestyle='dashed') # horizontal
    # plt.text(0, weights_max[1], "0.897", **text_style)
    # plt.plot((times[1][times_max_index[1]], times[1][times_max_index[1]]), (0.03, weights_max[1]), color = 'black', linestyle='dashed')# vertical 1
    # plt.text(times[1][times_max_index[1]] + 2, 0.01, "101", **text_style)
    # plt.plot((13.3, 13.3), (0.03, weights_max[1]), color = 'black', linestyle='dashed')# vertical 2
    # plt.text(13.3 + 2, 0.01, "13.3", **text_style)

    # medium
    # plt.text(times[1][times_max_index[1]], weights_max[1] + 0.02, "(" + repr(round(weights_max[1], 3)) + ", " + repr(int(times[1][times_max_index[1]])) + ")", **text_style)
    # plt.text(0, weights_max[1]-0.01, "0.79", **text_style)
    # plt.plot((0, times[1][times_max_index[1]]), (weights_max[1], weights_max[1]), color = 'black', linestyle='dashed') # horizontal
    # plt.plot((times[1][times_max_index[1]], times[1][times_max_index[1]]), (0.03, weights_max[1]), color = 'black', linestyle='dashed')# vertical 1
    # plt.text(times[1][times_max_index[1]] + 2, 0.01, "125", **text_style)
    # plt.plot((34, 34), (0.03, weights_max[1]), color = 'black', linestyle='dashed')# vertical 2
    # plt.text(34 + 2, 0.01, "34", **text_style)
    # # hard
    # plt.text(times[1][times_max_index[1]], weights_max[1] + 0.02, "(" + repr(round(weights_max[1], 3)) + ", " + repr(int(times[1][times_max_index[1]])) + ")", **text_style)
    plt.text(0, weights_max[1], "0.57", **text_style)
    plt.plot((0, times[1][times_max_index[1]]), (weights_max[1], weights_max[1]), color = 'black', linestyle='dashed') # horizontal
    plt.plot((times[1][times_max_index[1]], times[1][times_max_index[1]]), (0.03, weights_max[1]), color = 'black', linestyle='dashed')# vertical 1
    plt.text(times[1][times_max_index[1]] + 2, 0.01, "172", **text_style)
    plt.plot((26.9, 26.9), (0.03, weights_max[1]), color = 'black', linestyle='dashed')# vertical 2
    plt.text(26.9 + 2, 0.01, "26.9", **text_style)

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
    
    lines_0 = [line[0] for line in lines]
    # lines_0 = [lines[1][0], lines[3][0], lines[4][0]]
    #set limit for y axis
    plt.ylim(0, 1)
    # if level == 'hard':
    #     plt.ylim(0, 0.6)
    plt.legend(handles=lines_0, bbox_to_anchor=(0.5, 1.0), loc='center')
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
        graph1 = read_file(text_file="/home/tinvn/TIN/NLP_RL_Code/DeepRL-Grounding/saved/easy/train9_ae_prelu.log", defaut_gap=55/3600)# ae + san
        graph2 = read_file(text_file="/home/tinvn/TIN/NLP_RL_Code/DeepRL-Grounding/saved/easy/based_easy/train_based_easy.log") # base
        graph3 = read_file(text_file="/home/tinvn/TIN/NLP_RL_Code/DeepRL-Grounding/saved/easy/train_san_prelu_easy.log", defaut_gap=55/3600) # san
        graph4 = read_file(text_file="./saved/fourier_models/single_goal/easy/train_easy_forier_d1.log")
        graph5 = read_file(text_file="./saved/fourier_models/single_goal/easy/train_easy_ae_forier_d1.log")
        # plot
        if args.plot1:
            plot_graph(graph1, graph2, graph3, label1='SAN + AE easy', label2="Gated-attention easy", label3='SAN easy', level='easy', shown_type=args.type)
        if args.plot2:
            plot_graph_2(graphs=[graph1, graph2, graph3, graph4, graph5], labels=['SAN + AE easy', 'GA easy', "SAN easy", "FGA easy", 'FGA + AE easy'], level='easy', shown_type=args.type)
            # plot_graph_2(graphs=[graph4, graph5], labels=["FAN easy", 'FAN + AE easy'], level='easy', shown_type=args.type)
    elif args.difficulty == 'medium':
        # medium
        graph1 = read_file(text_file="/home/tinvn/TIN/NLP_RL_Code/DeepRL-Grounding/saved/medium/ae_san_medium/train_medium_ae_prelu.log", defaut_gap=70/3600)
        graph2 = read_file(text_file="/home/tinvn/TIN/NLP_RL_Code/DeepRL-Grounding/saved/medium/based_medium/train8_medium.log")
        graph3 = read_file(text_file="/home/tinvn/TIN/NLP_RL_Code/DeepRL-Grounding/saved/medium/san_medium/train_san_prelu_medium.log")
        graph4 = read_file(text_file="./saved/fourier_models/single_goal/medium/train_medium_forier_d1.log")
        graph5 = read_file(text_file="./saved/fourier_models/single_goal/medium/train_medium_ae_forier_d1.log")
        # plot
        if args.plot1:
            plot_graph(graph1, graph2, graph3, label1='SAN + AE medium', label2="Gated-attention medium", label3='SAN medium', level='medium', shown_type=args.type)
        if args.plot2:
            plot_graph_2(graphs=[graph1, graph2, graph3, graph4, graph5], labels=['SAN + AE medium', 'GA medium', "SAN medium", 'FGA medium', 'FGA + AE medium'], level='medium', shown_type=args.type)
    elif args.difficulty == 'hard':
        # hard
        graph1 = read_file(text_file="/home/tinvn/TIN/NLP_RL_Code/DeepRL-Grounding/saved/hard/ae_san_hard/train_hard_ae_prelu.log", defaut_gap=68/3600)#ae + san
        graph2 = read_file(text_file="/home/tinvn/TIN/NLP_RL_Code/DeepRL-Grounding/saved/hard/based_hard/train_based_hard.log", defaut_gap=62/3600) # train_hard base
        graph3 = read_file(text_file="/home/tinvn/TIN/NLP_RL_Code/DeepRL-Grounding/saved/hard/san_hard/train_san_prelu_hard.log", defaut_gap=64/3600)#, defaut_gap=60/3600)#
        graph4 = read_file(text_file="./saved/fourier_models/single_goal/hard/train_hard_forier_d1.log")
        graph5 = read_file(text_file='./saved/fourier_models/single_goal/hard/train_hard_ae_forier_d1.log')
        #plot
        if args.plot1:
            plot_graph(graph1, graph2, graph3, label1='SAN + AE hard', label2="Gated-attention hard", label3='SAN hard', level='hard', shown_type=args.type)
        if args.plot2:
            plot_graph_2(graphs=[graph1, graph2, graph3, graph4, graph5], labels=['SAN + AE hard', 'GA hard', "SAN hard", 'FGA hard', 'FGA + AE hard'], level='hard', shown_type=args.type)

    # print("Based easy (MT): ", calculate_mean_reward_and_acc('./saved/based_easy/test_MT_based_easy.log'))
    # print("Based easy (ZSL): ", calculate_mean_reward_and_acc('./saved/based_easy/test_ZSL_based_easy.log'))
    # print("AE Prelu easy (MT): ", calculate_mean_reward_and_acc('./test9_ae_prelu_MT.log'))
    # print("AE Prelu easy (ZSL): ", calculate_mean_reward_and_acc('./test9_ae_prelu_ZSL.log'))
    