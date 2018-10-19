import os
# for test on CPU
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"    #for Tensorflow cpu usage
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
from keras.models import Model
from keras.layers import Input, LSTM, Bidirectional
from keras.layers import Embedding, BatchNormalization
from keras.layers import Dense
import keras
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


# parameters of neural network
embed_size = 200 # dimension of embedding
max_seqs = 50 # max 50 words in a sequence
lstm_num = 200 # [50, 150, 200]
opt_lstm = 'RMSprop' # name of optimizer for lstm
dense_num = 100 # number of cells in the dense layer [50, 100, 150, 200]
dp_rate = 0.2 # dropout rate [0.1, 0.2, 0.3]
num_time = 3 # the number of time domains
num_label = 3 # the number of action labels
loss_senti = 'categorical_crossentropy'
loss_da = 'categorical_crossentropy'
l2_rate = 0.001 # l2 regularization rate
hp_lambda = 0.01 # flip gradient
ac_func = 'relu' # activation function ['relu', 'softplus', 'tanh']
epoch_num = 15
batch_size = 64

def da_lstm_topic(word_embed, da_weights, code_embed):
    """

    :param word_embed:
    :type word_embed: np.ndarray
    :param da_embed:
    :type da_embed: np.ndarray
    :return:
    """
    # inputs
    ctt_input = Input(shape=(max_seqs, ), dtype='int32', name='ctt_input')
    ctx_input = Input(shape=(max_seqs, ), dtype='int32', name='ctx_input')
    da_input = Input(shape=(max_seqs, ), dtype='int32', name='topic_input')
    da_ctx_input = Input(shape=(max_seqs, ), dtype='int32', name='da_ctx_input')
    code_input = Input(shape=(10, ), dtype='int32', name='code_input')

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
        da_weights.shape[0], da_weights.shape[1],
        weights=[da_weights], input_length=max_seqs, trainable=False,
        name='da_ctx_embed'
    )(da_ctx_input)
    
    code_embed = Embedding(
        code_embed.shape[0], code_embed.shape[1], # sizes
        weights=[code_embed], input_length=10, trainable=False,
        name='code_embed'
    )(code_input)

    # normalization
    ctt_norm = BatchNormalization(name='ctt_norm')(ctt_embed)
    da_norm = BatchNormalization(name='da_norm')(da_embed)
    

    # RNN, bi-LSTM
    ctt_lstm = Bidirectional(LSTM(lstm_num, dropout=dp_rate), name='ctt_lstm')(ctt_norm)
    ctx_lstm = Bidirectional(LSTM(lstm_num, dropout=dp_rate), name='ctx_lstm')(ctx_embed)
    da_lstm = Bidirectional(LSTM(lstm_num, dropout=dp_rate), name='da_lstm')(da_norm)
    da_ctx_lstm = Bidirectional(LSTM(lstm_num, dropout=dp_rate), name='da_ctx_lstm')(da_ctx_embed)
    code_lstm = Bidirectional(LSTM(lstm_num, dropout=dp_rate), name='code_lstm')(code_embed)

    ctt_merge = keras.layers.concatenate([ctx_lstm, ctt_lstm, code_lstm], axis=-1)
    da_merge = keras.layers.concatenate([da_ctx_lstm, da_lstm, code_lstm], axis=-1)

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
        inputs=[ctt_input, da_input, ctx_input, da_ctx_input, code_input],
        outputs=[preds_ctt, preds_da]
    )
    model.compile(
        loss={'preds_ctt': loss_senti, 'preds_da': loss_da},
        loss_weights={'preds_ctt': 1, 'preds_da': 0.01},
        optimizer=opt_lstm
    )

    print(model.summary())
    return model


def batch_indices_generator(data_len, batch_size):
    pass


"""
Load data
"""
data_path = './preprocessed_data/dataset.pkl'
data_whole = pickle.load(open(data_path, 'rb'))
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

#lda_tokenizer = Dictionary.load('./preprocessed_data/lda/lda_dict.pkl')
#lda_model = lda_helper.load_lda('./preprocessed_data/lda/lda.model')
#ctt_lda_idx = lda_helper.doc2idx(lda_tokenizer, data_whole['dataset_raw'], config.seq_max_len)
#ctx_lda_idx = lda_helper.doc2idx(lda_tokenizer, data_whole['context_raw'], config.seq_max_len)

ctt_lda_idx = data_whole['ctt_lda_idx']
ctx_lda_idx = data_whole['ctx_lda_idx']

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

# get labels
code_train = code_targets[train_indices]
code_valid = code_targets[valid_indices]
code_test = code_targets[test_indices]

da_train = data_whole['domain_labels'][train_indices]
da_valid = data_whole['domain_labels'][valid_indices]
da_test = data_whole['domain_labels'][test_indices]

# previous codes
prevs_train = processed_labels[train_indices]
prevs_valid = processed_labels[valid_indices]
prevs_test = processed_labels[test_indices]

print('Train Length: ' + str(len(train_indices)))
print('Valid Length: ' + str(len(valid_indices)))
print('Test Length: ' + str(len(test_indices)))
print('Overall MI Code Distribution: ' + str(Counter(codes)))
print('Training MI Code Distribution: ' + str(Counter(codes[train_indices])))
print('Validation MI Code Distribution: ' + str(Counter(codes[valid_indices])))
print('Testing MI Code Distribution: ' + str(Counter(codes[test_indices])))

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

if not os.path.exists('./weights/lda.npy'):
    lda_weights = lda_helper.init_weight('./preprocessed_data/lda/lda.model', 20)
    np.save('./weights/lda.npy', lda_weights)
else:
    lda_weights = np.load('./weights/lda.npy')

if not os.path.exists('./weights/code_prevs.npy'):
    import gensim
    label2vec_model = gensim.models.Word2Vec.load('./preprocessed_data/w2v_corpus/w2v_codes_50.txt')
    code_weights = np.zeros((len(labels_tokenizer.word_index) + 1, 50))
    for label_tmp, i in labels_tokenizer.word_index.items():
        # if word is found in the model, will be zeros
        if label_tmp in label2vec_model.wv:
            code_weights[i] = label2vec_model.wv.get_vector(label_tmp)
    np.save('./weights/code_prevs.npy', code_weights)
else:
    code_weights = np.load('./weights/code_prevs.npy')

"""
Build Model
"""
my_model = da_lstm_topic(embd_weights, lda_weights, code_weights)

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
        prevs_train_batch = prevs_train[i*batch_size:(i+1)*batch_size]
        # y
        code_train_batch = code_train[i*batch_size:(i+1)*batch_size]
        da_train_batch = da_train[i*batch_size:(i+1)*batch_size]
        
        tmp = my_model.train_on_batch(
            [
                ctt_train_batch, ctt_lda_train_batch, ctx_train_batch,
                ctx_lda_train_batch, prevs_train_batch
            ],
            {'preds_ctt': code_train_batch, 'preds_da': da_train_batch},
            class_weight={'preds_ctt:': 'auto', 'preds_da': 'auto'},
        )
        
        # calculate loss and accuracy
        loss += tmp[0]
        loss_avg = loss / (step)
        if step % 20 == 0:
            print('Step: {}'.format(step))
            print('\tLoss: {}.'.format(loss_avg))
            print('-------------------------------------------------')
        step += 1
        
    # valid on the validation set
    y_valid_preds = my_model.predict(
        [ctt_valid, ctt_lda_valid, ctx_valid, ctx_lda_valid, prevs_valid]
    )
    y_valid_preds = y_valid_preds[0]
    valid_f1 = f1_score(
        [np.argmax(item) for item in y_valid_preds], 
        [np.argmax(item) for item in code_valid],
        average='weighted'
    )
    
    if best_valid_f1 < valid_f1:
        """
        Test and record
        """
        best_valid_f1 = valid_f1
    print('Testing..............')
    y_pred = my_model.predict([ctt_test, ctt_lda_test, ctx_test, ctx_lda_test, prevs_test]);print()
    print(len(y_pred))
    print(len(y_pred[0]))
    y_pred = y_pred[0]

    test_f1 = f1_score(
        [np.argmax(item) for item in y_pred], 
        [np.argmax(item) for item in code_test],
        average='weighted')
    report = classification_report(
        [np.argmax(item) for item in y_pred],
        [np.argmax(item) for item in code_test]
    )
    print(report)

    # record
    with open('results_code.txt', 'a') as writefile:
        writefile.write('-----------------------------------\n')
        writefile.write('Valid F1: ' + str(best_valid_f1) + '\n')
        writefile.write('Weighted F1: ' + str(test_f1) + '\n')
        writefile.write(report)
