import os, sys

class SelfLogger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("./log/grid_ctt.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        pass

sys.stdout = SelfLogger()

import numpy as np
np.random.seed(7) # fix seed for reproductibility
import pickle

# self library
from utils import data_helper
from utils import extract_data_hdf5

# keras
from keras.layers import Dense, Dropout, Bidirectional
from keras.layers import GRU
from keras.layers.embeddings import Embedding
from keras.utils import np_utils
from keras.models import Sequential

# sklearn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, f1_score

# parameters for initial configuration
config = data_helper.BaseConfiguration()
config.MAX_SENTS = 10

from collections import Counter

"""data session"""
# check whether the json formatted dataset exist or now
dataset_path = '../preprocessed_data/dataset.pkl'
use_exist = False
if os.path.isfile(dataset_path):
    data_whole = pickle.load(open(dataset_path, 'rb'))
    use_exist = True

    dataset = data_whole['dataset_raw']
    contexts = data_whole['context_raw']
    labels = data_whole['labels']
    codes = data_whole['codes']

    processed_dataset = data_whole['dataset_processed']
    processed_contexts = data_whole['context_processed']
    processed_labels = data_whole['labels_preprocessed']
    code_targets = data_whole['codes_preprocessed']
    train_indices = data_whole['train_indices']
    valid_indices = data_whole['valid_indices']
    test_indices = data_whole['test_indices']
    keras_tokenizer = data_whole['keras_tokenizer']
    labels_tokenizer = data_whole['labels_tokenizer']

else:
    # load data
    data_filters = {'SPEAKER': {'P'}}
    data_filters['CODE'] = {'0+2', '0+3', '0+4', '0-2', '0-3', '0-4', 'C+2', 'C+3', 'C+4',
                                'C-4', 'FN', 'Fn', 'O+1', 'O+2', 'O+3', 'O+4', 'O+5', 'O-1', 'O-2',
                                'O-3', 'O-4', 'R+1', 'R+2', 'R+3', 'R+4', 'R-1', 'R-2', 'R-3',
                                'R-4', 'RA+2', 'RA+3', 'RA-2', 'RA-3', 'RD+3', 'RD+4', 'RD-3',
                                'RN+2', 'RN+3', 'Ra+2', 'Ra+3', 'Rd+3', 'Rn+3', 'TS+2', 'TS+3',
                                'TS+4', 'TS-2', 'Ts+2', 'c+1', 'c+2', 'c+3', 'c+4', 'c+5', 'c-1',
                                'c-2', 'c-3', 'c-4', 'f n', 'fn', 'o+1', 'o+2',
                                'o+3', 'o+4', 'o+5', 'o-1', 'o-2', 'o-3', 'o-4', 'o-5',
                                'r+1', 'r+2', 'r+3', 'r+4', 'r+5', 'r-1', 'r-2', 'r-3', 'r-4',
                                'r-5', 'ra+1', 'ra+2', 'ra+3', 'ra+4', 'ra+5', 'ra-1', 'ra-2',
                                'ra-3', 'ra-4', 'ra-5', 'rd+1', 'rd+2', 'rd+3', 'rd+4', 'rd+5',
                                'rd-2', 'rd-3', 'rd-4', 'rn+1', 'rn+2', 'rn+3',
                                'rn+4', 'rn-3', 'rn-4', 'rs+2', 'ts+1', 'st', 'ts+2', 'ts+3',
                                'ts+4', 'ts-1', 'ts-2', 'ts-3'}

    data_tuples = extract_data_hdf5.extract_hdf5("../MI_hdf5/*.hdf5", filters=data_filters, min_seqs=config.seq_min_len,
                                                 include_pre_contxt=True)
    dataset = []
    contexts = []
    labels = []
    codes = []

    for data, context, label, code in data_tuples:
        dataset.append(data)
        contexts.append(context)
        labels.append(label)
        codes.append(code)
    print('Whole dataset has ' + str(len(dataset)) + ' samples...............')

    # preprocess the data
    dataset = data_helper.preproc_data(dataset)
    contexts = data_helper.preproc_data(contexts)
    labels = np.asarray(labels)
    codes = np.asarray(data_helper.encode_codes(codes))  # encode the codes to -1, 0, 1

    all_dataset = dataset + contexts

    # in order to overcome the version problem
    if sys.version_info[0] == 2:
        from unidecode import unidecode

        processed_dataset, keras_tokenizer = data_helper.padding2sequences(
            [unidecode(item) for item in all_dataset], MAX_NB_WORDS=config.vocabulary_size,
            MAX_SEQUENCE_LENGTH=config.seq_max_len)
    else:
        processed_dataset, keras_tokenizer = data_helper.padding2sequences(
            all_dataset, MAX_NB_WORDS=config.vocabulary_size, MAX_SEQUENCE_LENGTH=config.seq_max_len)

    processed_labels, labels_tokenizer = data_helper.padding2sequences(
        [' '.join(items) for items in labels], MAX_NB_WORDS=config.vocabulary_size, MAX_SEQUENCE_LENGTH=10)
    processed_contexts = processed_dataset[len(dataset):]
    processed_dataset = processed_dataset[:len(dataset)]

    code_targets = np_utils.to_categorical(codes, num_classes=3)

    processed_dataset = np.asarray(processed_dataset)
    processed_contexts = np.asarray(processed_contexts)
    processed_labels = np.asarray(processed_labels)

    # split dataset into tvt
    train_indices, valid_indices, test_indices = data_helper.split_tvt_idx(processed_dataset)

    # save the data to json file for further references.
    data_whole = dict()
    data_whole['dataset_raw'] = dataset
    data_whole['context_raw'] = contexts
    data_whole['labels'] = labels
    data_whole['codes'] = codes

    data_whole['dataset_processed'] = processed_dataset
    data_whole['context_processed'] = processed_contexts
    data_whole['codes_preprocessed'] = code_targets
    data_whole['labels_preprocessed'] = processed_labels
    data_whole['train_indices'] = train_indices
    data_whole['valid_indices'] = valid_indices
    data_whole['test_indices'] = test_indices
    data_whole['keras_tokenizer'] = keras_tokenizer
    data_whole['labels_tokenizer'] = labels_tokenizer

    if sys.version_info[0] == 2:
        import codecs

        data_file = codecs.open(dataset_path, 'wb')
    else:
        data_file = open(dataset_path, 'wb')

    pickle.dump(data_whole, data_file, protocol=2)
    pickle.dump(keras_tokenizer, open("../preprocessed_data/keras_tokenizer.pkl", "wb"))
    pickle.dump(labels_tokenizer, open("../preprocessed_data/labels_tokenizer.pkl", "wb"))
    data_file.close()

x_train = processed_dataset[train_indices]
x_valid = processed_dataset[valid_indices]
x_test = processed_dataset[test_indices]

# get code of sentence
code_train = code_targets[train_indices]
code_valid = code_targets[valid_indices]
code_test = code_targets[test_indices]

print('Train Length: ' + str(len(train_indices)))
print('Valid Length: ' + str(len(valid_indices)))
print('Test Length: ' + str(len(test_indices)))
print('Overall MI Code Distribution: ' + str(Counter(codes)))
print('Training MI Code Distribution: ' + str(Counter(codes[train_indices])))
print('Validation MI Code Distribution: ' + str(Counter(codes[valid_indices])))
print('Testing MI Code Distribution: ' + str(Counter(codes[test_indices])))

best_model = None
best_score = 0.0
best_params = {}

# parameter settings
word2vec_paths = ['../preprocessed_data/w2v_corpus/glove.42B.300d.txt', '../preprocessed_data/w2v_corpus/w2v_tokenize.txt', '../preprocessed_data/w2v_corpus/google.bin', ]
ac_funcs = ['softplus', 'relu', 'tanh']
rnn_ac_funcs = ['tanh', 'hard_sigmoid']
dense_out_nums = [50, 100, 150, 200]
gru_out_nums = [50, 100, 150, 200]
dropout_rates = [0.1, 0.2]
epoch_nums = 10

print('Start to search parameters--------------------------------------------------')
for w2v_path in word2vec_paths:
    # load pretrained embeddings
    if 'glove' in w2v_path:
        word2vec_model = data_helper.load_glove(w2v_path)
    else:
        word2vec_model = data_helper.load_word2vec(w2v_path)
    config.embedding_size = len(word2vec_model['the'])

    embedding_matrix = np.zeros((len(keras_tokenizer.word_index) + 1, config.embedding_size))

    for word, i in keras_tokenizer.word_index.items():
        embeddings_vector = word2vec_model.get(word)
        # if word is found in the model, will be zeros
        if embeddings_vector is not None:
            embedding_matrix[i] = embeddings_vector

    for dense_num in dense_out_nums:
        for gru_num in gru_out_nums:
            for dp_rate in dropout_rates:
                for rnn_ac_func in rnn_ac_funcs:
                    for ac_func in ac_funcs:
                        test_model2 = Sequential()

                        embedding_layer_pretrained = Embedding(len(keras_tokenizer.word_index) + 1,
                                                               config.embedding_size,
                                                               weights=[embedding_matrix],
                                                               input_length=config.seq_max_len,
                                                               trainable=True,
                                                               name='content_embd')
                        print('Start to train')
                        test_model2.add(embedding_layer_pretrained)
                        test_model2.add(Bidirectional(GRU(gru_num, recurrent_dropout=dp_rate, kernel_initializer="glorot_uniform", recurrent_activation=rnn_ac_func),
                                               name='content_bilstm'))
                        test_model2.add(Dense(dense_num, activation=ac_func, kernel_initializer="glorot_uniform", name='final_nn'))
                        test_model2.add(Dropout(dp_rate))

                        test_model2.add(Dense(3, activation='sigmoid', name='final_output'))

                        # official document shows RMSprop is a better choice for recurrent neural network
                        test_model2.compile(loss='categorical_crossentropy', optimizer= 'rmsprop',
                                            metrics=['accuracy'])

                        print(test_model2.summary())

                        hist = test_model2.fit(x_train, code_train,
                                            epochs=epoch_nums,
                                            batch_size=64,
                                            validation_data=(x_valid, code_valid),
                                            class_weight='auto')

                        # multi label evaluation
                        best_valid_idx = np.argmax(hist.history['val_acc'])
                        best_valid_acc = hist.history['val_acc'][best_valid_idx]
                        best_valid_epc = hist.epoch[best_valid_idx]

                        # save the better model
                        if best_valid_acc > best_score:
                            best_model = test_model2
                            best_score = best_valid_acc

                            # save params
                            best_params['ac_func'] = ac_func
                            best_params['dropout_rate'] = dp_rate
                            best_params['gru_num'] = gru_num
                            best_params['dense_num'] = dense_num
                            best_params['w2v_path'] = w2v_path
                            best_params['best_valid_acc'] = best_valid_acc
                            best_params['rnn_ac_func'] = rnn_ac_func
                            best_params['best_valid_epc'] = best_valid_epc

                        print('Parameter Settings-----------------------------------------------')
                        current_params = dict()
                        current_params['ac_func'] = ac_func
                        current_params['dropout_rate'] = dp_rate
                        current_params['gru_num'] = gru_num
                        current_params['dense_num'] = dense_num
                        current_params['w2v_path'] = w2v_path
                        current_params['best_valid_acc'] = best_valid_acc
                        current_params['rnn_ac_func'] = rnn_ac_func
                        current_params['best_valid_epc'] = best_valid_epc
                        print(current_params)

best_model.save('../model/best_model_ctt.model')
print('------------------------------Testing-----------------------------------')
# multi label evaluation
y_pred = best_model.predict(x_test);print()
report = classification_report([np.argmax(item) for item in y_pred], [np.argmax(item) for item in code_test])
print(report)

print('F1_weighted score: ')
print(f1_score(y_pred=[np.argmax(item) for item in y_pred], y_true=[np.argmax(item) for item in code_test], average='weighted'))
print(confusion_matrix([np.argmax(item) for item in y_pred], [np.argmax(item) for item in code_test]))

best_params['best_test_report'] = str(report)
best_params['best_test_f1'] = str(f1_score(y_pred=[np.argmax(item) for item in y_pred], y_true=[np.argmax(item) for item in code_test], average='weighted'))
pickle.dump(best_params, open('../params/best_params_ctt.pkl', 'wb'))