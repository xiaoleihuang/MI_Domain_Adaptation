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


def label_change(dir_path, time_num = 3):
    if not os.path.exists('label_dist_count.pkl'):

        # create time sessions
        time_sessions = dict()
        # +, -, 0
        time_sessions[0] = np.asarray([0]*3) 
        time_sessions[1] = np.asarray([0]*3)
        time_sessions[2] = np.asarray([0]*3)
        time_sessions['general'] = np.asarray([0]*3)

        # loop through each file
        print('Reading each file in the directory: ' + dir_path)
        file_list = glob.glob(dir_path)
        for filep in file_list:
            print(filep)
            data = h5py.File(filep, 'r')

            # split into time sessions
            talk_len = len(data['CODE'])
            step = int(talk_len/3) + 1

            # loop through each line in the corpus
            for index, code in enumerate(data['CODE']):
                try:
                    code = code.decode('utf-8').strip()
                except UnicodeDecodeError:
                    print(index)
                    print(code)
                    continue
                
                key = int(index/step)
                if '-' in code:
                    label_idx = 1
                elif '+' in code:
                    label_idx = 0
                else:
                    label_idx = 2

                time_sessions[key][label_idx] += 1
                time_sessions['general'][label_idx] += 1
            
        # normalize the label distributions
        for key in time_sessions:
            time_sessions[key] = time_sessions[key] / sum(time_sessions[key])
            time_sessions[key] = dict(zip(range(len(time_sessions[key])), time_sessions[key]))
        # convert to data frame
        label_dist = pd.DataFrame.from_dict(time_sessions)
        pickle.dump(label_dist, open('label_dist_count.pkl', 'wb'))
    else:
        label_dist = pickle.load(open('label_dist_count.pkl', 'rb'))
    
    label_dist = label_dist.transpose()
    print(label_dist)
    # visualization
    label_dist.plot.bar(stacked=True, legend=True, colormap='Paired', fontsize=16, figsize=(7,7))
    plt.title('Intention Label Proportions Overtime', fontsize=16, y=1.02)
    plt.ylabel('Intention Label Proportion', fontsize=16)
    plt.ylim(0, 0.3)
    plt.xticks(list(range(4)), ['Stage 1', 'Stage 2', 'Stage 3', 'All Stages'], rotation= 0, fontsize=16)
    plt.xlabel('Time Stages', fontsize=16)

    plt.legend(loc=2, prop={'size': 16}, labels=['ct', 'st', 'fn'])
#    lg = plt.legend()
#    for idx, text in enumerate(['ct', 'st', 'fn']):
#        lg.get_texts()[idx].set_text(text)
    plt.savefig('./label_dist_count.pdf')

# run
label_change('../MI_hdf5/*.hdf5')
