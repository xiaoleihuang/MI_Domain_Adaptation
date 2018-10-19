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
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


def topic_change(dir_path, time_num = 3, ktopic=10):
    st = set(stopwords.words('english'))
    if not os.path.exists('topic_dist.pkl'):
        # load the dictionary and LDA model
        print('Loading models....................')
        lda_model = gensim.models.LdaModel.load('../preprocessed_data/lda/lda.model')
        lda_dict = pickle.load(open('../preprocessed_data/lda/lda_dict.pkl', 'rb'))


        # create time sessions
        time_sessions = dict()
        time_sessions[0] = np.asarray([0]*ktopic)
        time_sessions[1] = np.asarray([0]*ktopic)
        time_sessions[2] = np.asarray([0]*ktopic)
        time_sessions['general'] = np.asarray([0]*ktopic)
        
        # doc counts in each time session
        doc_counts = dict()
        doc_counts[0] = 0
        doc_counts[1] = 0
        doc_counts[2] = 0
        doc_counts['general'] = 0

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
                if data['SPEAKER'][index].decode('utf-8')!= 'P':
                    continue

                try:
                    line = line.decode('utf-8').strip()
                except UnicodeDecodeError:
                    continue
                
                
                if len(word_tokenize(line.strip())) < 5:
                    continue
                key = int(index/step)
                doc_counts[key] += 1
                doc_counts['general'] += 1

                topic_tmp = lda_model[lda_dict.doc2bow(word_tokenize(line.strip()))]
                topic_doc = [0.0]*lda_model.num_topics
                for tmp_pair in topic_tmp:
                    topic_doc[tmp_pair[0]] = tmp_pair[1]

                time_sessions[key] = np.add(time_sessions[key], topic_doc)
                time_sessions['general'] = np.add(time_sessions['general'], topic_doc)
            
        # normalize the topic distributions
        print(time_sessions)
        for key in time_sessions:
            time_sessions[key] = time_sessions[key] / doc_counts[key]
            time_sessions[key] = time_sessions[key] / sum(time_sessions[key])
            time_sessions[key] = dict(zip(range(len(time_sessions[key])), time_sessions[key]))
        # convert to data frame
        topic_dist = pd.DataFrame.from_dict(time_sessions)
        pickle.dump(topic_dist, open('topic_dist.pkl', 'wb'))
    else:
        topic_dist = pickle.load(open('topic_dist.pkl', 'rb'))
    
    topic_dist = topic_dist.transpose()
    print(topic_dist)
    # visualization
    csfont = {'fontname':'Times New Roman'}
    topic_dist.plot.bar(stacked=True, legend=False, colormap='tab20', fontsize=18, figsize=(7,7))
    plt.title('Topic Proportions Overtime', fontsize=20, **csfont)
    plt.ylabel('Topic Proportion', fontsize=18, **csfont)
    plt.xlabel('Time Stages', fontsize=18, **csfont)
    plt.xticks(list(range(4)), ['Stage 1', 'Stage 2', 'Stage 3', 'All Stages'], rotation=0, fontsize=16, **csfont)
    plt.savefig('./topic_dist1.pdf') # pdf


# run
topic_change('../MI_hdf5/*.hdf5', ktopic=10)
