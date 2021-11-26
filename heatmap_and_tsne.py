import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

import plotly
import plotly.graph_objs as go
from matplotlib import cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


root = "train_sentence_emb/ae/"
emb_file = open(root + "abc.txt", "r")
embs_string = []
lines = emb_file.readlines()

temp = ""
for i, emb in enumerate(lines):
    if "[" in emb:
        temp = ""
        temp += emb[1:]
    elif "]" in emb:
        temp += emb[:len(emb) - 2]
        embs_string.append(temp)
    else:
        temp += emb

embs_number = []
for emb in embs_string:
    temp = emb.split(" ")
    temp_list = []
    for i in temp:
        if i == '':
            continue
        if '\n' in i:
            temp_list.append(float(i[:-2]))
        else:
            temp_list.append(float(i))
    
    embs_number.append(temp_list)

# print(len(embs_number))
# print(len(embs_number[0]))


sentence_file = open(root + "sentence.txt", "r")
sentences = []
lines = sentence_file.readlines()
for i, sen in enumerate(lines):
    sentences.append(sen)

# print(len(sentences))

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
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
def levenstein(u, v):
    import jellyfish
    return jellyfish.levenshtein_distance(u, v)

sim_matrix = np.zeros((len(new_sentences), len(new_sentences)))
for i in range(len(new_sentences)):
    for j in range(len(new_sentences)):
        u = np.asarray(new_embs[i])
        v = np.asarray(new_embs[j])
        sim_matrix[i, j] = cosine(u, v)
        # sim_matrix[i, j] = levenstein(new_sentences[i], new_sentences[j])

import matplotlib.pyplot as plt

labels = []
for sent in new_sentences:
    labels.append(sent)

# fig, ax = plt.subplots(figsize=(20,20))
# cax = ax.matshow(sim_matrix, interpolation='nearest')
# ax.grid(True)

# ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
# plt.title('Instruction Vector Cosine Similarity matrix')
# plt.xticks(range(len(new_sentences)), labels, rotation=90, fontsize=7)
# plt.yticks(range(len(new_sentences)), labels, fontsize=7)
# plt.gcf().subplots_adjust(bottom=0.2)
# fig.colorbar(cax, ticks=[-0.39, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1])
# fig.colorbar(cax, ticks=[0.1, 2, 3, 4, 0.5, 0.6, 0.7, .75,.8,.85,.90,.95,17][::-1])
# plt.show()


# exit()
#sort by color:  blue, red,  green ,yellow
def sort_color(sentences, embs, color):
    new_sentences = []
    new_embs = []
    for i, (sent, emb) in enumerate(zip(sentences, embs)):
        if color in sent:
            new_sentences.append(sent)
            new_embs.append(emb)
    
    return new_sentences, new_embs

#-----------OBJECTS-------------------
blue_object_sentences, blue_object_embs = sort_color(objects_sentences, objects_embs, 'blue')
red_object_sentences, red_object_embs = sort_color(objects_sentences, objects_embs, 'red')
green_object_sentences, green_object_embs = sort_color(objects_sentences, objects_embs, 'green')
yellow_object_sentences, yellow_object_embs = sort_color(objects_sentences, objects_embs, 'yellow')

#objects by color
new_objects_sentences = []
new_objects_embs = []
for sent, emb in zip(blue_object_sentences, blue_object_embs):
    new_objects_sentences.append(sent)
    new_objects_embs.append(emb)
for sent, emb in zip(red_object_sentences, red_object_embs):
    new_objects_sentences.append(sent)
    new_objects_embs.append(emb)
for sent, emb in zip(green_object_sentences, green_object_embs):
    new_objects_sentences.append(sent)
    new_objects_embs.append(emb)
for sent, emb in zip(yellow_object_sentences, yellow_object_embs):
    new_objects_sentences.append(sent)
    new_objects_embs.append(emb)
#  remains are non color: smallest object, largest object, short object, tall object
for sent, emb in zip(objects_sentences, objects_embs):
    if 'red' not in sent and 'green' not in sent and 'yellow' not in sent and 'blue' not in sent:
        if "smallest object" in sent:
            new_objects_sentences.append(sent)
            new_objects_embs.append(emb)
        elif "largest object" in sent:
            new_objects_sentences.append(sent)
            new_objects_embs.append(emb)
        elif "short object" in sent:
            new_objects_sentences.append(sent)
            new_objects_embs.append(emb)
        elif "tall object" in sent:
            new_objects_sentences.append(sent)
            new_objects_embs.append(emb)

#-----------ARMORS-------------------
blue_armor_sentences, blue_armor_embs = sort_color(armors_sentences, armors_embs, 'blue')
red_armor_sentences, red_armor_embs = sort_color(armors_sentences, armors_embs, 'red')
green_armor_sentences, green_armor_embs = sort_color(armors_sentences, armors_embs, 'green')
yellow_armor_sentences, yellow_armor_embs = sort_color(armors_sentences, armors_embs, 'yellow')
#armor by color
new_armors_sentences = []
new_armors_embs = []
for sent, emb in zip(blue_armor_sentences, blue_armor_embs):
    new_armors_sentences.append(sent)
    new_armors_embs.append(emb)
for sent, emb in zip(red_armor_sentences, red_armor_embs):
    new_armors_sentences.append(sent)
    new_armors_embs.append(emb)
for sent, emb in zip(green_armor_sentences, green_armor_embs):
    new_armors_sentences.append(sent)
    new_armors_embs.append(emb)
for sent, emb in zip(yellow_armor_sentences, yellow_armor_embs):
    new_armors_sentences.append(sent)
    new_armors_embs.append(emb)
#  remains are non color: armor
for sent, emb in zip(armors_sentences, armors_embs):
    if 'red' not in sent and 'green' not in sent:
        if "armor" in sent:
            new_armors_sentences.append(sent)
            new_armors_embs.append(emb)

#-----------PILLARS-------------------
blue_pillar_sentences, blue_pillar_embs = sort_color(pillars_sentences, pillars_embs, 'blue')
red_pillar_sentences, red_pillar_embs = sort_color(pillars_sentences, pillars_embs, 'red')
green_pillar_sentences, green_pillar_embs = sort_color(pillars_sentences, pillars_embs, 'green')
yellow_pillar_sentences, yellow_pillar_embs = sort_color(pillars_sentences, pillars_embs, 'yellow')
#pillar by color
new_pillars_sentences = []
new_pillars_embs = []
for sent, emb in zip(blue_pillar_sentences, blue_pillar_embs):
    new_pillars_sentences.append(sent)
    new_pillars_embs.append(emb)
for sent, emb in zip(red_pillar_sentences, red_pillar_embs):
    new_pillars_sentences.append(sent)
    new_pillars_embs.append(emb)
for sent, emb in zip(green_pillar_sentences, green_pillar_embs):
    new_pillars_sentences.append(sent)
    new_pillars_embs.append(emb)
for sent, emb in zip(yellow_pillar_sentences, yellow_pillar_embs):
    new_pillars_sentences.append(sent)
    new_pillars_embs.append(emb)
#  remains are non color: pillar
for sent, emb in zip(pillars_sentences, pillars_embs):
    if 'red' not in sent and 'green' not in sent:
        if "pillar" in sent:
            new_pillars_sentences.append(sent)
            new_pillars_embs.append(emb)

#-----------TORCHs-------------------
blue_torch_sentences, blue_torch_embs = sort_color(torchs_sentences, torchs_embs, 'blue')
red_torch_sentences, red_torch_embs = sort_color(torchs_sentences, torchs_embs, 'red')
green_torch_sentences, green_torch_embs = sort_color(torchs_sentences, torchs_embs, 'green')
yellow_torch_sentences, yellow_torch_embs = sort_color(torchs_sentences, torchs_embs, 'yellow')
#torch by color
new_torchs_sentences = []
new_torchs_embs = []
for sent, emb in zip(blue_torch_sentences, blue_torch_embs):
    new_torchs_sentences.append(sent)
    new_torchs_embs.append(emb)
for sent, emb in zip(red_torch_sentences, red_torch_embs):
    new_torchs_sentences.append(sent)
    new_torchs_embs.append(emb)
for sent, emb in zip(green_torch_sentences, green_torch_embs):
    new_torchs_sentences.append(sent)
    new_torchs_embs.append(emb)
for sent, emb in zip(yellow_torch_sentences, yellow_torch_embs):
    new_torchs_sentences.append(sent)
    new_torchs_embs.append(emb)
#  remains are non color: torch
for sent, emb in zip(torchs_sentences, torchs_embs):
    if 'red' not in sent and 'green' not in sent and 'blue' not in sent:
        if "torch" in sent:
            new_torchs_sentences.append(sent)
            new_torchs_embs.append(emb)

#-----------KEYCARDS-------------------
blue_keycard_sentences, blue_keycard_embs = sort_color(keycards_sentences, keycards_embs, 'blue')
red_keycard_sentences, red_keycard_embs = sort_color(keycards_sentences, keycards_embs, 'red')
green_keycard_sentences, green_keycard_embs = sort_color(keycards_sentences, keycards_embs, 'green')
yellow_keycard_sentences, yellow_keycard_embs = sort_color(keycards_sentences, keycards_embs, 'yellow')
#keycard by color
new_keycards_sentences = []
new_keycards_embs = []
for sent, emb in zip(blue_keycard_sentences, blue_keycard_embs):
    new_keycards_sentences.append(sent)
    new_keycards_embs.append(emb)
for sent, emb in zip(red_keycard_sentences, red_keycard_embs):
    new_keycards_sentences.append(sent)
    new_keycards_embs.append(emb)
for sent, emb in zip(green_keycard_sentences, green_keycard_embs):
    new_keycards_sentences.append(sent)
    new_keycards_embs.append(emb)
for sent, emb in zip(yellow_keycard_sentences, yellow_keycard_embs):
    new_keycards_sentences.append(sent)
    new_keycards_embs.append(emb)
#  remains are non color: armor
for sent, emb in zip(keycards_sentences, keycards_embs):
    if 'red' not in sent and 'blue' not in sent and 'yellow' not in sent:
        if "keycard" in sent:
            new_keycards_sentences.append(sent)
            new_keycards_embs.append(emb)

#-----------SKULLKEYS-------------------
blue_skullkey_sentences, blue_skullkey_embs = sort_color(skullkey_sentences, skullkey_embs, 'blue')
red_skullkey_sentences, red_skullkey_embs = sort_color(skullkey_sentences, skullkey_embs, 'red')
green_skullkey_sentences, green_skullkey_embs = sort_color(skullkey_sentences, skullkey_embs, 'green')
yellow_skullkey_sentences, yellow_skullkey_embs = sort_color(skullkey_sentences, skullkey_embs, 'yellow')
#skullkey by colors
new_skullkeys_sentences = []
new_skullkeys_embs = []
for sent, emb in zip(blue_skullkey_sentences, blue_skullkey_embs):
    new_skullkeys_sentences.append(sent)
    new_skullkeys_embs.append(emb)
for sent, emb in zip(red_skullkey_sentences, red_skullkey_embs):
    new_skullkeys_sentences.append(sent)
    new_skullkeys_embs.append(emb)
for sent, emb in zip(green_skullkey_sentences, green_skullkey_embs):
    new_skullkeys_sentences.append(sent)
    new_skullkeys_embs.append(emb)
for sent, emb in zip(yellow_skullkey_sentences, yellow_skullkey_embs):
    new_skullkeys_sentences.append(sent)
    new_skullkeys_embs.append(emb)
#  remains are non color: skullkey
for sent, emb in zip(skullkey_sentences, skullkey_embs):
    if 'red' not in sent and 'blue' not in sent and 'yellow' not in sent:
        if "skullkey" in sent:
            new_skullkeys_sentences.append(sent)
            new_skullkeys_embs.append(emb)



new_sentences = []
new_embs = []
for sent, emb in zip(new_objects_sentences, new_objects_embs):
    new_sentences.append(sent)
    new_embs.append(emb)
for sent, emb in zip(new_armors_sentences, new_armors_embs):
    new_sentences.append(sent)
    new_embs.append(emb)
for sent, emb in zip(new_pillars_sentences, new_pillars_embs):
    new_sentences.append(sent)
    new_embs.append(emb)
for sent, emb in zip(new_torchs_sentences, new_torchs_embs):
    new_sentences.append(sent)
    new_embs.append(emb)
for sent, emb in zip(new_keycards_sentences, new_keycards_embs):
    new_sentences.append(sent)
    new_embs.append(emb)
for sent, emb in zip(new_skullkeys_sentences, new_skullkeys_embs):
    new_sentences.append(sent)
    new_embs.append(emb)

data = {'string': non_dup_sentences, 'embedding':non_dup_embs}
df = pd.DataFrame(data, columns = ['string', 'embedding'])
df.to_csv(root + 'data.csv', index=False)


def draw_attention_heatmap(sentences, embs):
    figure, ax = plt.subplots(figsize=(15, 256))
    array = np.array(embs, dtype=np.float64)

    im = ax.imshow(array)
    ax.set_xticks(np.arange(256))
    ax.set_yticks(np.arange(len(sentences)))
    ax.set_yticklabels(sentences)
    plt.setp(ax.get_yticklabels(), fontsize = 5, ha="right",
            rotation_mode="anchor")
    plt.setp(ax.get_xticklabels(), fontsize = 4,  ha="right",
            rotation_mode="anchor")

    plt.imshow(array, cmap='Blues_r', interpolation='nearest', aspect='auto')
    plt.show()


def draw_tsne_color(sentences, embs, dim=2):
    color = ['red', 'green', 'blue', 'yellow', 'unspecified']
    new_labels = []
    for sent, emb in zip(sentences, embs):
        has_color = False
        for c in color:
            if c in sent:
                new_labels.append(c)
                has_color = True
                break
        
        if has_color == False:
            new_labels.append('unspecified')
        
        has_color = False

    new_labels_indexes = []
    for label in new_labels:
        if label == 'red':
            new_labels_indexes.append(0)
        elif label == 'green':
            new_labels_indexes.append(1)
        elif label == 'blue':
            new_labels_indexes.append(2)
        elif label == 'yellow':
            new_labels_indexes.append(3)
        elif label == 'unspecified':
            new_labels_indexes.append(4)

    # new_labels_indexes = np.array(new_labels_indexes)

    tsne = TSNE(dim, verbose=1)
    tsne_proj = tsne.fit_transform(embs)

    tsne_proj_red = []
    tsne_proj_green = []
    tsne_proj_blue = []
    tsne_proj_yellow = []
    tsne_proj_non = []

    for label_index, emb in zip(new_labels_indexes, tsne_proj):
        if label_index == 0: #red
            tsne_proj_red.append(emb)
        elif label_index == 1: #green
            tsne_proj_green.append(emb)
        elif label_index == 2:#blue
            tsne_proj_blue.append(emb)
        elif label_index == 3:
            tsne_proj_yellow.append(emb)
        else:
            tsne_proj_non.append(emb)

    tsne_proj_red = np.array(tsne_proj_red)
    tsne_proj_green = np.array(tsne_proj_green)
    tsne_proj_blue = np.array(tsne_proj_blue)
    tsne_proj_yellow = np.array(tsne_proj_yellow)
    tsne_proj_non = np.array(tsne_proj_non)

    cmap = cm.get_cmap('tab20')
    data=[]

    if dim == 2:
        trace_red = go.Scatter(
                    x = tsne_proj_red[:,0], 
                    y = tsne_proj_red[:,1],  
                    name = 'red',
                    textposition = "top center",
                    textfont_size = 20,
                    mode = 'markers+text',
                    marker = {
                        'size': 10,
                        'opacity': 0.8,
                        'color': 'red'
                    }

                )

        trace_green = go.Scatter(
                    x = tsne_proj_green[:,0], 
                    y = tsne_proj_green[:,1],
                    name = 'green',
                    textposition = "top center",
                    textfont_size = 20,
                    mode = 'markers+text',
                    marker = {
                        'size': 10,
                        'opacity': 0.8,
                        'color': 'green'
                    }

                )

        trace_blue = go.Scatter(
                    x = tsne_proj_blue[:,0], 
                    y = tsne_proj_blue[:,1],  
                    name = 'blue',
                    textposition = "top center",
                    textfont_size = 20,
                    mode = 'markers+text',
                    marker = {
                        'size': 10,
                        'opacity': 0.8,
                        'color': 'blue'
                    }

                )

        trace_yellow = go.Scatter(
                    x = tsne_proj_yellow[:,0], 
                    y = tsne_proj_yellow[:,1],
                    name = 'yellow',
                    textposition = "top center",
                    textfont_size = 20,
                    mode = 'markers+text',
                    marker = {
                        'size': 10,
                        'opacity': 0.8,
                        'color': 'yellow'
                    }

                )

        trace_non = go.Scatter(
                    x = tsne_proj_non[:,0], 
                    y = tsne_proj_non[:,1], 
                    name = 'Unspecified Color',
                    textposition = "top center",
                    textfont_size = 20,
                    mode = 'markers+text',
                    marker = {
                        'size': 10,
                        'opacity': 0.8,
                        'color': 'black'
                    }

                )
    elif dim == 3:
        trace_red = go.Scatter3d(
                    x = tsne_proj_red[:,0], 
                    y = tsne_proj_red[:,1],  
                    z = tsne_proj_red[:,2],
                    name = 'red',
                    textposition = "top center",
                    textfont_size = 20,
                    mode = 'markers+text',
                    marker = {
                        'size': 10,
                        'opacity': 0.8,
                        'color': 'red'
                    }

                )

        trace_green = go.Scatter3d(
                    x = tsne_proj_green[:,0], 
                    y = tsne_proj_green[:,1],
                    z = tsne_proj_green[:,2],
                    name = 'green',
                    textposition = "top center",
                    textfont_size = 20,
                    mode = 'markers+text',
                    marker = {
                        'size': 10,
                        'opacity': 0.8,
                        'color': 'green'
                    }

                )

        trace_blue = go.Scatter3d(
                    x = tsne_proj_blue[:,0], 
                    y = tsne_proj_blue[:,1],  
                    z = tsne_proj_blue[:,2],
                    name = 'blue',
                    textposition = "top center",
                    textfont_size = 20,
                    mode = 'markers+text',
                    marker = {
                        'size': 10,
                        'opacity': 0.8,
                        'color': 'blue'
                    }

                )

        trace_yellow = go.Scatter3d(
                    x = tsne_proj_yellow[:,0], 
                    y = tsne_proj_yellow[:,1],
                    z = tsne_proj_yellow[:,2],
                    name = 'yellow',
                    textposition = "top center",
                    textfont_size = 20,
                    mode = 'markers+text',
                    marker = {
                        'size': 10,
                        'opacity': 0.8,
                        'color': 'yellow'
                    }

                )

        trace_non = go.Scatter3d(
                    x = tsne_proj_non[:,0], 
                    y = tsne_proj_non[:,1], 
                    z = tsne_proj_non[:,2],
                    name = 'Unspecified Color',
                    textposition = "top center",
                    textfont_size = 20,
                    mode = 'markers+text',
                    marker = {
                        'size': 10,
                        'opacity': 0.8,
                        'color': 'black'
                    }

                )
    else:
        print("Wrong dimension input !!!!")

    data.append(trace_red)
    data.append(trace_green)
    data.append(trace_blue)
    data.append(trace_yellow)
    data.append(trace_non)

    # Configure the layout

    layout = go.Layout(
        margin = {'l': 0, 'r': 0, 'b': 0, 't': 0},
        showlegend=True,
        legend=dict(
        x=1,
        y=0.5,
        font=dict(
            family="Courier New",
            size=25,
            color="black"
        )),
        font = dict(
            family = " Courier New ",
            size = 15),
        autosize = False,
        width = 1000,
        height = 1000
        )

    plot_figure = go.Figure(data = data, layout = layout)
    plot_figure.show()
    ax.legend(fontsize='large', markerscale=2)
    plt.show()

def draw_tsne_object(sentences, embs, dim = 2):
    objects = ['Armor', 'Pillar', 'Torch', 'Keycard', 'Skullkey', 'unspecified']

    new_labels = []
    for sent, emb in zip(sentences, embs):
        has_object = False
        for ob in objects:
            if ob.lower() in sent.lower():
                new_labels.append(ob)
                has_object = True
                break
        
        if has_object == False:
            new_labels.append('unspecified')
        
        has_object = False

    new_labels_indexes = []
    for label in new_labels:
        if label == 'Armor':
            new_labels_indexes.append(0)
        elif label == 'Pillar':
            new_labels_indexes.append(1)
        elif label == 'Torch':
            new_labels_indexes.append(2)
        elif label == 'Keycard':
            new_labels_indexes.append(3)
        elif label == 'Skullkey':
            new_labels_indexes.append(4)
        else:
            new_labels_indexes.append(5)

    # new_labels_indexes = np.array(new_labels_indexes)

    tsne = TSNE(dim, verbose=1)
    tsne_proj = tsne.fit_transform(embs)

    tsne_proj_armor = []
    tsne_proj_pillar = []
    tsne_proj_torch = []
    tsne_proj_keycard = []
    tsne_proj_skullkey = []
    tsne_proj_non = []

    for label_index, emb in zip(new_labels_indexes, tsne_proj):
        if label_index == 0:
            tsne_proj_armor.append(emb)
        elif label_index == 1:
            tsne_proj_pillar.append(emb)
        elif label_index == 2:
            tsne_proj_torch.append(emb)
        elif label_index == 3:
            tsne_proj_keycard.append(emb)
        elif label_index == 4:
            tsne_proj_skullkey.append(emb)
        else:
            tsne_proj_non.append(emb)

    tsne_proj_armor = np.array(tsne_proj_armor)
    tsne_proj_pillar = np.array(tsne_proj_pillar)
    tsne_proj_torch = np.array(tsne_proj_torch)
    tsne_proj_keycard = np.array(tsne_proj_keycard)
    tsne_proj_skullkey = np.array(tsne_proj_skullkey)
    tsne_proj_non = np.array(tsne_proj_non)

    cmap = cm.get_cmap('tab20')
    data=[]

    if dim == 2:
        trace_armor = go.Scatter(
                    x = tsne_proj_armor[:,0], 
                    y = tsne_proj_armor[:,1],  
                    name = 'Armor',
                    textposition = "top center",
                    textfont_size = 20,
                    mode = 'markers+text',
                    marker = {
                        'size': 10,
                        'opacity': 0.8,
                        'color': 'red'
                    }

                )

        trace_pillar = go.Scatter(
                    x = tsne_proj_pillar[:,0], 
                    y = tsne_proj_pillar[:,1],
                    name = 'Pillar',
                    textposition = "top center",
                    textfont_size = 20,
                    mode = 'markers+text',
                    marker = {
                        'size': 10,
                        'opacity': 0.8,
                        'color': 'green'
                    }

                )

        trace_torch = go.Scatter(
                    x = tsne_proj_torch[:,0], 
                    y = tsne_proj_torch[:,1],  
                    name = 'Torch',
                    textposition = "top center",
                    textfont_size = 20,
                    mode = 'markers+text',
                    marker = {
                        'size': 10,
                        'opacity': 0.8,
                        'color': 'blue'
                    }

                )

        trace_keycard = go.Scatter(
                    x = tsne_proj_keycard[:,0], 
                    y = tsne_proj_keycard[:,1],
                    name = 'Keycard',
                    textposition = "top center",
                    textfont_size = 20,
                    mode = 'markers+text',
                    marker = {
                        'size': 10,
                        'opacity': 0.8,
                        'color': 'yellow'
                    }

                )

        trace_skullkey = go.Scatter(
                    x = tsne_proj_skullkey[:,0], 
                    y = tsne_proj_skullkey[:,1], 
                    name = 'Skullkey',
                    textposition = "top center",
                    textfont_size = 20,
                    mode = 'markers+text',
                    marker = {
                        'size': 10,
                        'opacity': 0.8,
                        'color': 'purple'
                    }

                )

        trace_non = go.Scatter(
                    x = tsne_proj_non[:,0], 
                    y = tsne_proj_non[:,1], 
                    name = 'Unspecified Object',
                    textposition = "top center",
                    textfont_size = 20,
                    mode = 'markers+text',
                    marker = {
                        'size': 10,
                        'opacity': 0.8,
                        'color': 'black'
                    }

                )
    elif dim == 3:
        trace_armor = go.Scatter3d(
                    x = tsne_proj_armor[:,0], 
                    y = tsne_proj_armor[:,1],  
                    z = tsne_proj_armor[:,2],
                    name = 'Armor',
                    textposition = "top center",
                    textfont_size = 20,
                    mode = 'markers+text',
                    marker = {
                        'size': 10,
                        'opacity': 0.8,
                        'color': 'red'
                    }

                )

        trace_pillar = go.Scatter3d(
                    x = tsne_proj_pillar[:,0], 
                    y = tsne_proj_pillar[:,1],
                    z = tsne_proj_pillar[:,2],
                    name = 'Pillar',
                    textposition = "top center",
                    textfont_size = 20,
                    mode = 'markers+text',
                    marker = {
                        'size': 10,
                        'opacity': 0.8,
                        'color': 'green'
                    }

                )

        trace_torch = go.Scatter3d(
                    x = tsne_proj_torch[:,0], 
                    y = tsne_proj_torch[:,1],  
                    z = tsne_proj_torch[:,2],
                    name = 'Torch',
                    textposition = "top center",
                    textfont_size = 20,
                    mode = 'markers+text',
                    marker = {
                        'size': 10,
                        'opacity': 0.8,
                        'color': 'blue'
                    }

                )

        trace_keycard = go.Scatter3d(
                    x = tsne_proj_keycard[:,0], 
                    y = tsne_proj_keycard[:,1],
                    z = tsne_proj_keycard[:,2],
                    name = 'Keycard',
                    textposition = "top center",
                    textfont_size = 20,
                    mode = 'markers+text',
                    marker = {
                        'size': 10,
                        'opacity': 0.8,
                        'color': 'yellow'
                    }

                )

        trace_skullkey = go.Scatter3d(
                    x = tsne_proj_skullkey[:,0], 
                    y = tsne_proj_skullkey[:,1], 
                    z = tsne_proj_skullkey[:,2],
                    name = 'Skullkey',
                    textposition = "top center",
                    textfont_size = 20,
                    mode = 'markers+text',
                    marker = {
                        'size': 10,
                        'opacity': 0.8,
                        'color': 'purple'
                    }

                )
        
        trace_non = go.Scatter3d(
                    x = tsne_proj_non[:,0], 
                    y = tsne_proj_non[:,1], 
                    z = tsne_proj_non[:,2],
                    name = 'Unspecified Object',
                    textposition = "top center",
                    textfont_size = 20,
                    mode = 'markers+text',
                    marker = {
                        'size': 10,
                        'opacity': 0.8,
                        'color': 'black'
                    }

                )

    else:
        print("Wrong dimension input !!!")

    data.append(trace_armor)
    data.append(trace_pillar)
    data.append(trace_torch)
    data.append(trace_keycard)
    data.append(trace_skullkey)
    data.append(trace_non)

    # Configure the layout

    layout = go.Layout(
        margin = {'l': 0, 'r': 0, 'b': 0, 't': 0},
        showlegend=True,
        legend=dict(
        x=1,
        y=0.5,
        font=dict(
            family="Courier New",
            size=25,
            color="black"
        )),
        font = dict(
            family = " Courier New ",
            size = 15),
        autosize = False,
        width = 1000,
        height = 1000
        )

    plot_figure = go.Figure(data = data, layout = layout)
    plot_figure.show()
    ax.legend(fontsize='large', markerscale=2)
    plt.show()

def draw_tsne_size(sentences, embs, dim = 2):
    size = ['largest', 'smallest', 'tall', 'short', 'unspecified']
    new_labels = []
    for sent, emb in zip(sentences, embs):
        has_size = False
        for s in size:
            if s in sent:
                new_labels.append(s)
                has_size = True
                break
        
        if has_size == False:
            new_labels.append('unspecified')
        
        has_size = False

    new_labels_indexes = []
    for label in new_labels:
        if label == 'largest':
            new_labels_indexes.append(0)
        elif label == 'smallest':
            new_labels_indexes.append(1)
        elif label == 'tall':
            new_labels_indexes.append(2)
        elif label == 'short':
            new_labels_indexes.append(3)
        elif label == 'unspecified':
            new_labels_indexes.append(4)

    # new_labels_indexes = np.array(new_labels_indexes)

    tsne = TSNE(dim, verbose=1)
    tsne_proj = tsne.fit_transform(embs)

    tsne_proj_largest = []
    tsne_proj_smallest = []
    tsne_proj_tall = []
    tsne_proj_short = []
    tsne_proj_non = []

    for label_index, emb in zip(new_labels_indexes, tsne_proj):
        if label_index == 0: 
            tsne_proj_largest.append(emb)
        elif label_index == 1: 
            tsne_proj_smallest.append(emb)
        elif label_index == 2:
            tsne_proj_tall.append(emb)
        elif label_index == 3:
            tsne_proj_short.append(emb)
        else:
            tsne_proj_non.append(emb)

    tsne_proj_largest = np.array(tsne_proj_largest)
    tsne_proj_smallest = np.array(tsne_proj_smallest)
    tsne_proj_tall = np.array(tsne_proj_tall)
    tsne_proj_short = np.array(tsne_proj_short)
    tsne_proj_non = np.array(tsne_proj_non)

    cmap = cm.get_cmap('tab20')
    data=[]

    if dim == 2:
        trace_largest = go.Scatter(
                    x = tsne_proj_largest[:,0], 
                    y = tsne_proj_largest[:,1],  
                    name = 'Largest',
                    textposition = "top center",
                    textfont_size = 20,
                    mode = 'markers+text',
                    marker = {
                        'size': 10,
                        'opacity': 0.8,
                        'color': 'red'
                    }

                )

        trace_smallest = go.Scatter(
                    x = tsne_proj_smallest[:,0], 
                    y = tsne_proj_smallest[:,1],
                    name = 'Smallest',
                    textposition = "top center",
                    textfont_size = 20,
                    mode = 'markers+text',
                    marker = {
                        'size': 10,
                        'opacity': 0.8,
                        'color': 'green'
                    }

                )

        trace_tall = go.Scatter(
                    x = tsne_proj_tall[:,0], 
                    y = tsne_proj_tall[:,1],  
                    name = 'Tall',
                    textposition = "top center",
                    textfont_size = 20,
                    mode = 'markers+text',
                    marker = {
                        'size': 10,
                        'opacity': 0.8,
                        'color': 'blue'
                    }

                )

        trace_short = go.Scatter(
                    x = tsne_proj_short[:,0], 
                    y = tsne_proj_short[:,1],
                    name = 'Short',
                    textposition = "top center",
                    textfont_size = 20,
                    mode = 'markers+text',
                    marker = {
                        'size': 10,
                        'opacity': 0.8,
                        'color': 'yellow'
                    }

                )

        trace_non = go.Scatter(
                    x = tsne_proj_non[:,0], 
                    y = tsne_proj_non[:,1], 
                    name = 'Unspecified Size',
                    textposition = "top center",
                    textfont_size = 20,
                    mode = 'markers+text',
                    marker = {
                        'size': 10,
                        'opacity': 0.8,
                        'color': 'black'
                    }

                )
    elif dim == 3:
        trace_largest = go.Scatter3d(
                    x = tsne_proj_largest[:,0], 
                    y = tsne_proj_largest[:,1],  
                    z = tsne_proj_largest[:,2],
                    name = 'Largest',
                    textposition = "top center",
                    textfont_size = 20,
                    mode = 'markers+text',
                    marker = {
                        'size': 10,
                        'opacity': 0.8,
                        'color': 'red'
                    }

                )

        trace_smallest = go.Scatter3d(
                    x = tsne_proj_smallest[:,0], 
                    y = tsne_proj_smallest[:,1],
                    z = tsne_proj_smallest[:,2],
                    name = 'Smallest',
                    textposition = "top center",
                    textfont_size = 20,
                    mode = 'markers+text',
                    marker = {
                        'size': 10,
                        'opacity': 0.8,
                        'color': 'green'
                    }

                )

        trace_tall = go.Scatter3d(
                    x = tsne_proj_tall[:,0], 
                    y = tsne_proj_tall[:,1],  
                    z = tsne_proj_tall[:,2],
                    name = 'Tall',
                    textposition = "top center",
                    textfont_size = 20,
                    mode = 'markers+text',
                    marker = {
                        'size': 10,
                        'opacity': 0.8,
                        'color': 'blue'
                    }
                )

        trace_short = go.Scatter3d(
                    x = tsne_proj_short[:,0], 
                    y = tsne_proj_short[:,1],
                    z = tsne_proj_short[:,2],
                    name = 'Short',
                    textposition = "top center",
                    textfont_size = 20,
                    mode = 'markers+text',
                    marker = {
                        'size': 10,
                        'opacity': 0.8,
                        'color': 'yellow'
                    }

                )

        trace_non = go.Scatter3d(
                    x = tsne_proj_non[:,0], 
                    y = tsne_proj_non[:,1], 
                    z = tsne_proj_non[:,2],
                    name = 'Unspecified Size',
                    textposition = "top center",
                    textfont_size = 20,
                    mode = 'markers+text',
                    marker = {
                        'size': 10,
                        'opacity': 0.8,
                        'color': 'black'
                    }

                )
    else:
        print("Wrong dimension input !!!")

    data.append(trace_largest)
    data.append(trace_smallest)
    data.append(trace_tall)
    data.append(trace_short)
    data.append(trace_non)

    # Configure the layout

    layout = go.Layout(
        margin = {'l': 0, 'r': 0, 'b': 0, 't': 0},
        showlegend=True,
        legend=dict(
        x=1,
        y=0.5,
        font=dict(
            family="Courier New",
            size=25,
            color="black"
        )),
        font = dict(
            family = " Courier New ",
            size = 15),
        autosize = False,
        width = 1000,
        height = 1000
        )

    plot_figure = go.Figure(data = data, layout = layout)
    plot_figure.show()
    # ax.legend(fontsize='large', markerscale=2)
    plt.show()


# draw_tsne_color(new_sentences, new_embs)
# draw_tsne_object(new_sentences, new_embs)
# draw_tsne_size(new_sentences, new_embs, dim = 2)

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))

def draw_pca(sentences, embs, type='color'):
    sents = np.asarray(embs)
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(sents)

    fig, ax = plt.subplots()
    if type == 'color':
        for i, s in enumerate(sentences):
            if 'yellow' in s:
                ax.scatter(x=pca_results[i,0], y=pca_results[i,1],color='yellow',label='yellow')
            elif 'red' in s:
                ax.scatter(x=pca_results[i,0], y=pca_results[i,1],color='red',label='red')
            elif 'blue' in s:
                ax.scatter(x=pca_results[i,0], y=pca_results[i,1],color='blue',label='blue')
            elif 'green' in s:
                ax.scatter(x=pca_results[i,0], y=pca_results[i,1],color='green',label='green')
            else:
                ax.scatter(x=pca_results[i,0], y=pca_results[i,1],color='black',label='unspecified')
    
    if type == 'size':
        for i, s in enumerate(sentences):
            if 'largest' in s:
                ax.scatter(x=pca_results[i,0], y=pca_results[i,1],color='red',label='largest')
            elif 'smallest' in s:
                ax.scatter(x=pca_results[i,0], y=pca_results[i,1],color='blue',label='smallest')
            elif 'tall' in s:
                ax.scatter(x=pca_results[i,0], y=pca_results[i,1],color='green',label='tall')
            elif 'short' in s:
                ax.scatter(x=pca_results[i,0], y=pca_results[i,1],color='yellow',label='short')
            else:
                ax.scatter(x=pca_results[i,0], y=pca_results[i,1],color='black',label='unspecified')

    if type == 'object':
        for i, s in enumerate(sentences):
            if 'armor' in s:
                ax.scatter(x=pca_results[i,0], y=pca_results[i,1],color='red',label='Armor')
            elif 'pillar' in s:
                ax.scatter(x=pca_results[i,0], y=pca_results[i,1],color='blue',label='Pillar')
            elif 'torch' in s:
                ax.scatter(x=pca_results[i,0], y=pca_results[i,1],color='green',label='Torch')
            elif 'keycard' in s:
                ax.scatter(x=pca_results[i,0], y=pca_results[i,1],color='yellow',label='Keycard')
            elif 'skullkey' in s:
                ax.scatter(x=pca_results[i,0], y=pca_results[i,1],color='purple',label='Skullkey')
            else:
                ax.scatter(x=pca_results[i,0], y=pca_results[i,1],color='black',label='unspecified')


    legend_without_duplicate_labels(ax)
    plt.show()

def draw_tsne2(sentences, embs, type='color'):
    sents = np.asarray(embs)
    tsne = TSNE(n_components=2, n_iter=2000)
    tsne_results = tsne.fit_transform(sents)

    def plot_tsne_color(tsne_results, sentences):
        fig, ax = plt.subplots()
        for i, s in enumerate(sentences):
            if 'yellow' in s:
                ax.scatter(x=tsne_results[i,0], y=tsne_results[i,1],color='yellow',label='yellow', marker="x")
            elif 'red' in s:
                ax.scatter(x=tsne_results[i,0], y=tsne_results[i,1],color='red',label='red', marker="D")
            elif 'blue' in s:
                ax.scatter(x=tsne_results[i,0], y=tsne_results[i,1],color='blue',label='blue', marker="*")
            elif 'green' in s:
                ax.scatter(x=tsne_results[i,0], y=tsne_results[i,1],color='green',label='green', marker="s")
            else:
                ax.scatter(x=tsne_results[i,0], y=tsne_results[i,1],color='black',label='unspecified')
        legend_without_duplicate_labels(ax)
        plt.show()
    
    def plot_tsne_size(tsne_results, sentences):
        fig, ax = plt.subplots()
        for i, s in enumerate(sentences):
            if 'largest' in s:
                ax.scatter(x=tsne_results[i,0], y=tsne_results[i,1],color='red',label='largest', marker="D")
            elif 'smallest' in s:
                ax.scatter(x=tsne_results[i,0], y=tsne_results[i,1],color='blue',label='smallest', marker="*")
            elif 'tall' in s:
                ax.scatter(x=tsne_results[i,0], y=tsne_results[i,1],color='green',label='tall', marker="s")
            elif 'short' in s:
                ax.scatter(x=tsne_results[i,0], y=tsne_results[i,1],color='yellow',label='short', marker="x")
            else:
                ax.scatter(x=tsne_results[i,0], y=tsne_results[i,1],color='black',label='unspecified')
        legend_without_duplicate_labels(ax)
        plt.show()

    def plot_tsne_object(tsne_results, sentences):
        fig, ax = plt.subplots()
        for i, s in enumerate(sentences):
            if 'armor' in s:
                ax.scatter(x=tsne_results[i,0], y=tsne_results[i,1],color='red',label='Armor', marker="D")
            elif 'pillar' in s:
                ax.scatter(x=tsne_results[i,0], y=tsne_results[i,1],color='blue',label='Pillar', marker="*")
            elif 'torch' in s:
                ax.scatter(x=tsne_results[i,0], y=tsne_results[i,1],color='green',label='Torch', marker="s")
            elif 'keycard' in s:
                ax.scatter(x=tsne_results[i,0], y=tsne_results[i,1],color='yellow',label='Keycard', marker="x")
            elif 'skullkey' in s:
                ax.scatter(x=tsne_results[i,0], y=tsne_results[i,1],color='purple',label='Skullkey', marker="v")
            else:
                ax.scatter(x=tsne_results[i,0], y=tsne_results[i,1],color='black',label='unspecified')
        legend_without_duplicate_labels(ax)
        plt.show()

    if type == 'color':
        plot_tsne_color(tsne_results, sentences)

    if type == 'size':
        plot_tsne_size(tsne_results, sentences)

    if type == 'object':
        plot_tsne_object(tsne_results, sentences)
    
    else:
        plot_tsne_color(tsne_results, sentences)
        plot_tsne_size(tsne_results, sentences)
        plot_tsne_object(tsne_results, sentences)

type = 'all'
# draw_pca(new_sentences, new_embs, type='color')
# draw_pca(new_sentences, new_embs, type='size')
# draw_pca(new_sentences, new_embs, type='object')
draw_tsne2(new_sentences, new_embs, type)


