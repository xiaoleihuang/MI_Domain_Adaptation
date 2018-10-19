import glob
import h5py
import os
from os.path import basename
import pickle
import numpy as np
import gensim
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def topic_change(dir_path, time_num = 3):
    if not os.path.exists('topic_dist_count.pkl'):
        # load the dictionary and LDA model
        print('Loading models....................')
        lda_model = gensim.models.LdaModel.load('../preprocessed_data/lda/lda.model')
        lda_dict = pickle.load(open('../preprocessed_data/lda/lda_dict.pkl', 'rb'))

        # create time sessions
        time_sessions = dict()
        time_sessions[0] = np.asarray([0]*20)
        time_sessions[1] = np.asarray([0]*20)
        time_sessions[2] = np.asarray([0]*20)
        time_sessions['general'] = np.asarray([0]*20)

        # loop through each file
        print('Reading each file in the directory: ' + dir_path)
        file_list = glob.glob(dir_path)
        for filep in file_list:
            print(filep)
            data = h5py.File(filep, 'r')

            # split into time sessions
            talk_len = len(data['WORDS'])
            step = int(talk_len/3) + 1

            # loop through each line in the corpus
            for index, line in enumerate(data['WORDS']):
                try:
                    line = line.decode('utf-8').strip()
                except UnicodeDecodeError:
                    print(index)
                    print(line)
                    continue
                
                key = int(index/step)

                topic_tmp = lda_model[lda_dict.doc2bow(line.strip().split())]
                topic_doc = [0.0]*lda_model.num_topics
                for tmp_pair in topic_tmp:
                    topic_doc[tmp_pair[0]] = tmp_pair[1]
                topic_doc = np.asarray(topic_doc)
                topest_label = topic_doc.argsort()[0]

                time_sessions[key][topest_label] = time_sessions[key][topest_label] + 1
                time_sessions['general'][topest_label] = time_sessions['general'][topest_label] + 1
            
        # normalize the topic distributions
        for key in time_sessions:
            time_sessions[key] = time_sessions[key] / sum(time_sessions[key])
            time_sessions[key] = dict(zip(range(len(time_sessions[key])), time_sessions[key]))
        # convert to data frame
        topic_dist = pd.DataFrame.from_dict(time_sessions)
        pickle.dump(topic_dist, open('topic_dist_count.pkl', 'wb'))
    else:
        topic_dist = pickle.load(open('topic_dist_count.pkl', 'rb'))
    
    topic_dist = topic_dist.transpose()
    print(topic_dist)
    # visualization
    topic_dist.plot.bar(stacked=True, legend=False, colormap='Paired', fontsize=20, figsize=(7,7))
    plt.title('Topic Change Over Time Sessions', fontsize=20)
    plt.ylabel('Topic Proportion', fontsize=20)
    plt.xticks(list(range(4)), ['Session 1', 'Session 2', 'Session 3', 'All Sessions'], rotation=20, fontsize=20)
    plt.savefig('./topic_dist_count.pdf')


# run
topic_change('../MI_hdf5/*.hdf5')
