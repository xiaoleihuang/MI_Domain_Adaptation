"""
Single GPU
"""

import os

import sys
from keras.models import Model
from keras.layers import Input, LSTM, Bidirectional
from keras.layers import Embedding, BatchNormalization
from keras.layers import Dense
import keras
import keras.backend as K
import numpy as np
from utils import flipGradientTF
import pickle
from gensim.corpora import Dictionary
from utils import lda_helper, data_helper, model_helper
from collections import Counter
# parameters for initial configuration
config = data_helper.BaseConfiguration()
config.MAX_SENTS = 10
# evaluation metrics
from sklearn.metrics import classification_report, f1_score

record_my = open('record_my.txt', 'w')
record_xiao2016 = open('record_xiao2016.txt', 'w')

# shared parameters
embed_size = 200 # dimension of embedding
max_seqs = 50 # max 50 words in a sequence
epoch_num = 10
batch_size = 64
num_label = 3 # the number of action labels
loss_senti = 'categorical_crossentropy'

def get_devices():
    """Get a list of available GPUS"""
    devices = [name.name for name in K.get_session().list_devices() if 'GPU' in name.name]
    devices = ['/' + ':'.join(name.lower().replace('/', '').split(':')[-2:]) for name in devices]
    return devices


def xiao2016(word_embed):
    """
    Model of the Xiao2016 paper
    :param word_embed:
    :type word_embed: np.ndarray
    :param da_embed:
    :type da_embed: np.ndarray
    :return:
    """
    # inputs
    ctt_input = Input(shape=(max_seqs, ), dtype='int32', name='ctt_input')
    ctx_input = Input(shape=(max_seqs, ), dtype='int32', name='ctx_input')

    # embedding
    ctt_embed = Embedding(
        word_embed.shape[0], word_embed.shape[1], # sizes
        weights=[word_embed], input_length=max_seqs, trainable=True,
        name='ctt_embed'
    )(ctt_input)

    ctx_embed = Embedding(
        word_embed.shape[0], word_embed.shape[1], # sizes
        weights=[word_embed], input_length=max_seqs, trainable=True,
        name='ctx_embed'
    )(ctx_input)

    # normalization
    ctt_norm = BatchNormalization(name='ctt_norm')(ctt_embed)
    
    # RNN, bi-LSTM
    ctt_lstm = Bidirectional(LSTM(256, dropout=.2), name='ctt_lstm')(ctt_norm)
    ctx_lstm = Bidirectional(LSTM(256, dropout=.2), name='ctx_lstm')(ctx_embed)
    
    ctt_merge = keras.layers.concatenate([ctx_lstm, ctt_lstm], axis=-1)

    # define sentiment classification
    dense_ctt = Dense(256, activation='relu', name='dense_ctt')(ctt_merge)
    preds_ctt = Dense(num_label, activation='softmax', name='preds_ctt')(dense_ctt)

    model = Model(
        inputs=[ctt_input, ctx_input],
        outputs=preds_ctt
    )
    model.compile(loss=loss_senti, optimizer= 'adam', metrics=['accuracy'])
    print(model.summary())
    return model


def da_lstm_topic(word_embed, da_weights):
    """

    :param word_embed:
    :type word_embed: np.ndarray
    :param da_embed:
    :type da_embed: np.ndarray
    :return:
    """
    # parameter seetings for the proposed model
    lstm_num = 200 # [50, 150, 200]
    opt_lstm = 'RMSprop' # name of optimizer for lstm
    dense_num = 100 # number of cells in the dense layer [50, 100, 150, 200]
    dp_rate = 0.2 # dropout rate [0.1, 0.2, 0.3]
    num_time = 3 # the number of time domains
    loss_da = 'categorical_crossentropy'
    l2_rate = 0.001 # l2 regularization rate
    hp_lambda = 0.05 # flip gradient [0.05, 0.01, 0.005]
    ac_func = 'softplus' # activation function ['relu', 'softplus', 'tanh']
    da_loss_weight = 0.05 # [0.1, 0.05, 0.01, 0.005]
    
    # inputs
    ctt_input = Input(shape=(max_seqs, ), dtype='int32', name='ctt_input')
    ctx_input = Input(shape=(max_seqs, ), dtype='int32', name='ctx_input')
    da_input = Input(shape=(50, ), dtype='int32', name='topic_input')
    da_ctx_input = Input(shape=(50, ), dtype='int32', name='topic_ctx_input')

    # embedding
    ctt_embed = Embedding(
        word_embed.shape[0], word_embed.shape[1], # sizes
        weights=[word_embed], input_length=max_seqs, trainable=True,
        name='ctt_embed'
    )(ctt_input)

    ctx_embed = Embedding(
        word_embed.shape[0], word_embed.shape[1], # sizes
        weights=[word_embed], input_length=max_seqs, trainable=True,
        name='ctx_embed'
    )(ctx_input)

    da_embed = Embedding(
        da_weights.shape[0], da_weights.shape[1],
        weights=[da_weights], input_length=max_seqs, trainable=False,
        name='da_embed'
    )(da_input)

    da_ctx_embed = Embedding(
        da_weights.shape[0], da_weights.shape[1], weights=[da_weights], 
        input_length=max_seqs, trainable=False,
        name='da_ctx_embed'
    )(da_ctx_input)

    # normalization
    ctt_norm = BatchNormalization(name='ctt_norm')(ctt_embed)
    da_norm = BatchNormalization(name='da_norm')(da_embed)
    

    # RNN, bi-LSTM
    ctt_lstm = Bidirectional(LSTM(lstm_num, dropout=dp_rate), name='ctt_lstm')(ctt_norm)
    ctx_lstm = Bidirectional(LSTM(lstm_num, dropout=dp_rate), name='ctx_lstm')(ctx_embed)
    da_lstm = Bidirectional(LSTM(lstm_num, dropout=dp_rate), name='da_lstm')(da_norm)
    da_ctx_lstm = Bidirectional(LSTM(lstm_num, dropout=dp_rate), name='da_ctx_lstm')(da_ctx_embed)
    
    ctt_merge = keras.layers.concatenate([ctx_lstm, ctt_lstm], axis=-1)
    da_merge = keras.layers.concatenate([da_ctx_lstm, da_lstm], axis=-1)

    # define task for domain classification
    flip = flipGradientTF.GradientReversal(hp_lambda, name='flip')(da_merge) # flip gradient
    dense_da = Dense(dense_num, activation='relu', name='dense_da')(flip)
    preds_da = Dense(num_time, activation='softmax', name='preds_da')(dense_da)

    # Merge
    merge_inputs = keras.layers.concatenate([ctt_merge, da_merge], axis=-1)

    # define sentiment classification
    dense_ctt = Dense(dense_num, activation=ac_func, name='dense_ctt')(merge_inputs)
    preds_ctt = Dense(num_label, activation='softmax', name='preds_ctt')(dense_ctt)

    model = Model(
        inputs=[ctt_input, da_input, ctx_input, da_ctx_input],
        outputs=[preds_ctt, preds_da]
    )
    model.compile(
        loss={'preds_ctt': loss_senti, 'preds_da': loss_da},
        loss_weights={'preds_ctt': 1, 'preds_da': da_loss_weight},
        optimizer=opt_lstm
    )

    print(model.summary())
    return model


# number of bootstrap samples
b_num = 100
# generate 100 numbers for the sample seeds
np.random.seed(33)
b_seeds = np.random.randint(1000, size=b_num)

# get available GPU devices
available_devices = get_devices()
print('Available devices: ' + '; '.join(available_devices))

"""
Load data
"""
print('Loading data')
data_path = './preprocessed_data/dataset.pkl'
data_whole = pickle.load(open(data_path, 'rb'))
dataset = data_whole['dataset_raw']
contexts = data_whole['context_raw']
labels = data_whole['labels']
codes = data_whole['codes']
da_labels = data_whole['domain_labels']

processed_dataset = data_whole['dataset_processed']
processed_contexts = data_whole['context_processed']
processed_labels = data_whole['labels_preprocessed']
keras_tokenizer = data_whole['keras_tokenizer']
labels_tokenizer = data_whole['labels_tokenizer']
code_targets = data_whole['codes_preprocessed']

lda_tokenizer = Dictionary.load('./preprocessed_data/lda/lda_dict.pkl')
lda_model = lda_helper.load_lda('./preprocessed_data/lda/lda.model')
ctt_lda_idx = lda_helper.doc2idx(lda_tokenizer, data_whole['dataset_raw'], config.seq_max_len)
ctx_lda_idx = lda_helper.doc2idx(lda_tokenizer, data_whole['context_raw'], config.seq_max_len)

data_length = len(processed_dataset)

"""
Initialize weights
"""
print('Initialize weights')
if not os.path.exists('./weights/w2v.npy'):
    # load pretrained embeddings
    w2v_path = './preprocessed_data/w2v_corpus/google.bin'
    if 'glove' in w2v_path:
        w2v_model = data_helper.load_glove(w2v_path)
    else:
        w2v_model = data_helper.load_word2vec(w2v_path)
    config.embedding_size = len(w2v_model['the'])
    # initialize weights
    embd_weights = model_helper.init_weights(w2v_model, keras_tokenizer, config.embedding_size)
    np.save('./weights/w2v.npy', embd_weights)
else:
    embd_weights = np.load('./weights/w2v.npy')

# load embedding for the xiao2016, glovec
if not os.path.exists('./weights/glove.npy'):
    # load pretrained embeddings
    w2v_path = './preprocessed_data/w2v_corpus/glove300.txt'
    if 'glove' in w2v_path:
        w2v_model = data_helper.load_glove(w2v_path)
    else:
        w2v_model = data_helper.load_word2vec(w2v_path)
    config.embedding_size = len(w2v_model['the'])
    # initialize weights
    embd_weights_xiao = model_helper.init_weights(w2v_model, keras_tokenizer, config.embedding_size)
    np.save('./weights/glove.npy', embd_weights)
else:
    embd_weights_xiao = np.load('./weights/glove.npy')

if not os.path.exists('./weights/lda.npy'):
    lda_weights = lda_helper.init_weight('./preprocessed_data/lda/lda.model', 20)
    np.save('./weights/lda.npy', lda_weights)
else:
    lda_weights = np.load('./weights/lda.npy')

"""
Run bootstrap sample process
"""
train_indices_bp = data_whole['train_indices']
valid_indices_bp = data_whole['valid_indices']
test_indices_bp = data_whole['test_indices']

for round_idx, rand_seed in enumerate(b_seeds):
    print('Bootstrap Round: ' + str(round_idx))
    # sample train_indices; valid_indices; test_indices
    #np.random.seed(rand_seed)
    train_indices = np.random.randint(len(train_indices_bp), size=len(train_indices_bp))
    valid_indices = np.random.randint(len(valid_indices_bp), size=len(train_indices_bp))
    test_indices = np.random.randint(len(test_indices_bp), size=len(train_indices_bp))

    train_indices = train_indices_bp[train_indices]
    valid_indices = valid_indices_bp[valid_indices]
    test_indices = test_indices_bp[test_indices]

    # get training data
    ctt_train = processed_dataset[train_indices]
    ctt_valid = processed_dataset[valid_indices]
    ctt_test = processed_dataset[test_indices]

    ctt_lda_train = ctt_lda_idx[train_indices]
    ctt_lda_valid = ctt_lda_idx[valid_indices]
    ctt_lda_test = ctt_lda_idx[test_indices]

    ctx_train = processed_contexts[train_indices]
    ctx_valid = processed_contexts[valid_indices]
    ctx_test = processed_contexts[test_indices]

    ctx_lda_train = ctx_lda_idx[train_indices]
    ctx_lda_valid = ctx_lda_idx[valid_indices]
    ctx_lda_test = ctx_lda_idx[test_indices]

    # get true labels
    code_train = code_targets[train_indices]
    code_valid = code_targets[valid_indices]
    code_test = code_targets[test_indices]

    da_train = data_whole['domain_labels'][train_indices]
    da_valid = data_whole['domain_labels'][valid_indices]
    da_test = data_whole['domain_labels'][test_indices]

    """
    Build Model
    """
    my_model = da_lstm_topic(embd_weights, lda_weights)
    xiao2016_model = xiao2016(embd_weights_xiao)

    """
    Training process
    """
    print('Training process................')
    data_len = len(ctt_train)
    if len(ctt_train) % batch_size == 0:
        steps = int(len(ctt_train) / batch_size)
    else:
        steps = int(len(ctt_train) / batch_size) + 1
    best_valid_f1 = 0.0
    best_valid_f1_xiao2016 = 0.0
    
    test_f1 = 0.0
    test_f1_xiao2016 = 0.0

    for e in range(epoch_num):
        loss = 0.0
        step = 1
        print('--------------Epoch: {}--------------'.format(e))
        for i in range(steps):
            # x
            ctt_train_batch = ctt_train[i*batch_size:(i+1)*batch_size]
            ctt_lda_train_batch = ctt_lda_train[i*batch_size:(i+1)*batch_size]
            ctx_train_batch = ctx_train[i*batch_size:(i+1)*batch_size]
            ctx_lda_train_batch = ctx_lda_train[i*batch_size:(i+1)*batch_size]
            # y
            code_train_batch = code_train[i*batch_size:(i+1)*batch_size]
            da_train_batch = da_train[i*batch_size:(i+1)*batch_size]

            my_model.train_on_batch(
                [ctt_train_batch, ctt_lda_train_batch, ctx_train_batch, ctx_lda_train_batch],
                {'preds_ctt': code_train_batch, 'preds_da': da_train_batch},
                class_weight={'preds_ctt:': 'auto', 'preds_da': 'auto'},
            )
            xiao2016_model.train_on_batch([ctt_train_batch, ctx_train_batch], code_train_batch, class_weight='auto')
            
        """
        Validation process
        """
        print('Validation.............')
        # our model
        y_valid_preds = my_model.predict([ctt_valid, ctt_lda_valid, ctx_valid, ctx_lda_valid])
        y_valid_preds = y_valid_preds[0]
        valid_f1 = f1_score(
            [np.argmax(item) for item in y_valid_preds], 
            [np.argmax(item) for item in code_valid],
            average='weighted'
        )

        # xiao2016
        y_valid_preds_xiao = xiao2016_model.predict([ctt_valid, ctx_valid])
        #y_valid_preds = y_valid_preds[0]
        valid_f1_xiao = f1_score(
            [np.argmax(item) for item in y_valid_preds_xiao], 
            [np.argmax(item) for item in code_valid],
            average='weighted'
        )
        
        """
        Test process
        """
        if best_valid_f1 < valid_f1:
            best_valid_f1 = valid_f1
            print('Testing..............')
            y_pred = my_model.predict([ctt_test, ctt_lda_test, ctx_test, ctx_lda_test]);print()
            y_pred = y_pred[0]

            test_f1 = f1_score(
                [np.argmax(item) for item in y_pred], 
                [np.argmax(item) for item in code_test],
                average='weighted')
            
            print(classification_report(
                [np.argmax(item) for item in y_pred],
                [np.argmax(item) for item in code_test]
            ))

        if best_valid_f1_xiao2016 < valid_f1_xiao:
            best_valid_f1_xiao2016 = valid_f1_xiao
            y_pred = xiao2016_model.predict([ctt_test, ctx_test]);print()
            #y_pred = y_pred[0]

            test_f1_xiao = f1_score(
                [np.argmax(item) for item in y_pred], 
                [np.argmax(item) for item in code_test],
                average='weighted')
            
            print(classification_report(
                [np.argmax(item) for item in y_pred],
                [np.argmax(item) for item in code_test]
            ))
    
    # record
    print('Our Model best valid f1: %.3f, test f1: %.3f' % (best_valid_f1, test_f1))
    print('Xiao2016 best valid f1: %.3f, test f1: %.3f' % (best_valid_f1_xiao2016, test_f1_xiao))
    record_my.write(str(test_f1) + '\n')
    record_my.flush()
    record_xiao2016.write(str(test_f1_xiao) + '\n')
    record_xiao2016.flush()

record_my.close()
record_xiao2016.close()
