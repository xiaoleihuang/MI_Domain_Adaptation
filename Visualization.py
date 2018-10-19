
# coding: utf-8

# In[102]:

"""
1st Visualize embedding of each sentences; compare with word2vec
"""
import numpy as np
np.random.seed(7) # fix seed for reproductibility
import json

# dimention reduction
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# visualization
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from utils import data_helper

# Deep Learning tools
import tensorflow as tf
from keras.models import load_model


# In[50]:

def plot_embedding(X, labels, label_names={1:'positive', 0:'neutral', -1:'negative'}, n_components=3, mode='tsne'):
    """This function is to conduct dimension reduction & visualize the embedding of inputs (sentences, or documents).
    The dimension number must be 2 or 3.
    
    Args:
        X (list): a list of sentences or documents embeddings
        labels (list): a list of labels corresponds to X
        label_names (dict): a dictionary of mapping between labels and their names
        n_components (int): 2 or 3 dimensions
        mode (str): dimension reduction method, currently support only tsne and pca
    """
    # check dimensions
    if n_components not in [2, 3]:
        print('The dimension number must be 2 or 3.')
        return
    
    # check mode
    if mode == 'tsne':
        model_tsne = TSNE(n_components=n_components)
        X_new = model_tsne.fit_transform(X)
    elif mode == 'pca':
        model_pca = PCA(n_components=n_components)
        X_new = model_pca.fit_transform(X)
    else:
        print('The input mode, ' + str(mode) + ' is not supported!')
        return
    
    # check labels and label_names
    if len(np.unique(labels)) != len(label_names):
        print('The <labels> parameter must share the same size with the <label_names>!')
        return
    
    # visualization
    plt.figure(figsize=(10, 10))
    colors = ['navy', 'turquoise', 'darkorange'][:n_components]
    
    for color, label_tmp, label_name in zip(colors, labels, np.unique(labels)):
        if n_components == 3:
            plt.scatter(X_new[labels==label_tmp, 0], X_new[labels==label_tmp, 1],
                X_new[labels==label_tmp, 2],
                    color=color, lw=2, label=label_name)
        else:
            plt.scatter(X_new[labels==label_tmp, 0], X_new[labels==label_tmp, 1],
                   color=color, lw=2, label=label_name)
    
    plt.title(str(n_components) + 'D' + ' embedding visualization by ' + mode.upper())
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.show()


# In[33]:

datafile = open('./viz/data/embedding_info.json')
dataset = json.load(datafile)    


# In[40]:

sent_embedding = list(data_helper.cal_embedding(dataset['test_sents_raw'], './model/rnn_model (copy).model', dataset['word_indx']))


# In[80]:

model_tsne = TSNE(n_components=3)
X_new = model_tsne.fit_transform(sent_embedding)


fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
colors = ['red', 'black', 'blue']
for idx, labelsss in enumerate(dataset['test_labels']):
    if labelsss == 0:
        ax.scatter(X_new[idx, 0], X_new[idx,1], X_new[idx,2], color=colors[0], lw=2)
    elif labelsss == 1:
        ax.scatter(X_new[idx, 0], X_new[idx,1], X_new[idx,2], color=colors[1], lw=2)
    else:
        ax.scatter(X_new[idx, 0], X_new[idx,1], X_new[idx,2], color=colors[-1], lw=2)

# ax.title(str(3) + 'D' + ' embedding visualization by ' + 'tsne'.upper())
# ax.legend(loc="best")
plt.show()


# In[46]:

# plot_embedding(sent_embedding, dataset['test_labels'])

