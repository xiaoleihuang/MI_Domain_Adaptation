from utils import extract_data_hdf5
from utils import data_helper
import sys
import keras
from keras.utils import np_utils
import numpy as np
import pickle

def build_dataset(output='./preprocessed_data/dataset.pkl'):
    # parameters for initial configuration
    config = data_helper.BaseConfiguration()
    config.MAX_SENTS = 10

    # load data
    data_filters = {'SPEAKER': set(['P'])}
    data_filters['CODE'] = set(['0+2', '0+3', '0+4', '0-2', '0-3', '0-4', 'C+2', 'C+3', 'C+4',
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
        'ts+4', 'ts-1', 'ts-2', 'ts-3'])

    data_tuples = extract_data_hdf5.extract_hdf5("./MI_hdf5/*.hdf5", filters=data_filters, min_seqs=config.seq_min_len,
                                                 include_pre_contxt=True)
    dataset = []
    contexts = []
    labels = []
    codes = []
    domain_labels = []

    for data, context, label, code, da_label in data_tuples:
        dataset.append(data)
        contexts.append(context)
        labels.append(label)
        codes.append(code)
        domain_labels.append(da_label)
    print('Whole dataset has ' + str(len(dataset)) + ' samples...............')

    # preprocess the data
    dataset = data_helper.preproc_data(dataset)
    contexts = data_helper.preproc_data(contexts)
    labels = np.asarray(labels)
    codes = np.asarray(data_helper.encode_codes(codes))  # encode the codes to -1, 0, 1
    domain_labels = np.asarray(domain_labels)

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

    if keras.__version__.startswith('1'):
        code_targets = np_utils.to_categorical(codes, nb_classes=3)
    else:
        code_targets = np_utils.to_categorical(codes, num_classes=3)

    processed_dataset = np.asarray(processed_dataset)
    processed_contexts = np.asarray(processed_contexts)
    processed_labels = np.asarray(processed_labels)

    train_indices, valid_indices, test_indices = data_helper.split_tvt_idx(processed_dataset)

    # save the data to json file for further references.
    data_whole = dict()
    data_whole['dataset_raw'] = dataset
    data_whole['context_raw'] = contexts
    data_whole['labels'] = labels
    data_whole['codes'] = codes
    data_whole['domain_labels'] = domain_labels

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
        data_file = codecs.open(output, 'wb')
    else:
        data_file = open(output, 'wb')

    pickle.dump(data_whole, data_file, protocol=2)
    pickle.dump(keras_tokenizer, open("./preprocessed_data/keras_tokenizer.pkl", "wb"))
    pickle.dump(labels_tokenizer, open("./preprocessed_data/labels_tokenizer.pkl", "wb"))
    data_file.close()

build_dataset()
