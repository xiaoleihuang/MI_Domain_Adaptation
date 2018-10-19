"""This script is to train a model for the Web-App with extra self-defined documents.
Process:
1. load original dataset
2. load the extra dataset (self-defined)
3. Merge the two datasets
4. Train three models: ctt only; ctt+ctx; ctt+ctx
5. Save them as temporary files.
"""

import numpy as np
np.random.seed(7) # fix seed for reproductibility
import sys, os, pickle
os.environ['KERAS_BACKEND']='theano'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from keras.layers import Dense, Dropout, Bidirectional
from keras.layers import GRU, Input, Layer
from keras.layers.embeddings import Embedding
from keras.utils import np_utils
from keras.models import Model, Sequential

# self library
import data_helper, extract_data_hdf5
# parameters for initial configuration
config = data_helper.BaseConfiguration()
config.MAX_SENTS = 10

from collections import Counter

import keras.initializers as initializations
import keras.backend as K
import keras
class Attention(Layer):
    def __init__(self,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializations.get('glorot_uniform')

        self.W_regularizer = keras.regularizers.get(W_regularizer)
        self.b_regularizer = keras.regularizers.get(b_regularizer)

        self.W_constraint = keras.constraints.get(W_constraint)
        self.b_constraint = keras.constraints.get(b_constraint)

        self.bias = bias
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer=self.init,
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True
        super(Attention, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        eij = K.dot(x, self.W)

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def get_output_shape_for(self, input_shape):
        return input_shape[0], input_shape[-1]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

"""1. load orginal dataset"""

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

"""2. load the extra dataset (self-defined)"""
extra_ctt, extra_ctx, extra_labels = data_helper.add_extra_docs('extra_train_docs.tsv')

"""3. Merge into one"""
dataset.extend(extra_ctt)
contexts.extend(extra_ctx)
# labels are the previous 10 label
label_extra = ['unknown'] * 10
for idx in range(len(extra_ctt)):
    labels.append(label_extra)

"""4. Preprocessing
No splitting into train/valid/test, because this is only to train models for Web_Apps
"""
# preprocess the data
dataset = data_helper.preproc_data(dataset)
contexts = data_helper.preproc_data(contexts)
labels = np.asarray(labels)
codes = data_helper.encode_codes(codes)  # encode the codes to -1, 0, 1
codes.extend(extra_labels) # merge the codes here
codes = np.asarray(codes)

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

label2vec_model = data_helper.load_word2vec('../preprocessed_data/w2v_corpus/w2v_codes.txt')
print('Overall MI Code Distribution: ' + str(Counter(codes)))
print('Train Length: ' + str(len(processed_dataset)))


"""Parameters"""
gru_num = 100
dp_rate = 0.2
dense_num = 100
ac_func = 'softplus'
l2_rate = 0.0005
rnn_ac_func = 'tanh'
epoch_nums = 10

word2vec_paths = '../preprocessed_data/w2v_corpus/google.bin'
word2vec_model = data_helper.load_word2vec(word2vec_paths)
config.embedding_size = len(word2vec_model['the'])
embedding_matrix = np.zeros((len(keras_tokenizer.word_index) + 1, config.embedding_size))

for word, i in keras_tokenizer.word_index.items():
    embeddings_vector = word2vec_model.get(word)
    # if word is found in the model, will be zeros
    if embeddings_vector is not None:
        embedding_matrix[i] = embeddings_vector

embedding_layer_pretrained = Embedding(len(keras_tokenizer.word_index) + 1,
        config.embedding_size,
        weights=[embedding_matrix],
        input_length=config.seq_max_len,
        trainable=True,
        name='content_embd')
embedding_layer_pretrained_context = Embedding(
        len(keras_tokenizer.word_index) + 1, config.embedding_size,
        weights=[embedding_matrix], input_length=config.seq_max_len,
        trainable=True, name='context_embd')

print('Start to compile model:')
sent_input_contxt = Input(shape=(config.seq_max_len,), dtype='int32',
                          name='context_input')
embedded_sequences_contxt = embedding_layer_pretrained_context(sent_input_contxt)
l_lstm_contxt = Bidirectional(GRU(gru_num, return_sequences=True, recurrent_dropout=dp_rate, kernel_initializer="glorot_uniform", recurrent_activation=rnn_ac_func),
                              name='context_bilstm')(embedded_sequences_contxt)
l_att_contxt = Attention(W_regularizer=keras.regularizers.l1_l2(0, l2_rate),
                         name='context_att')(l_lstm_contxt)
l_att_contxt_do = Dropout(dp_rate)(l_att_contxt)

sent_input = Input(shape=(config.seq_max_len,), dtype='int32', name='content_input')
embedded_sequences = embedding_layer_pretrained(sent_input)
l_lstm = Bidirectional(GRU(gru_num, return_sequences=True, recurrent_dropout=dp_rate, kernel_initializer="glorot_uniform", recurrent_activation=rnn_ac_func),
                       name='content_bilstm')(embedded_sequences)
l_att = Attention(W_regularizer=keras.regularizers.l1_l2(0, l2_rate),
                  name='content_att')(l_lstm)
l_att_do = Dropout(dp_rate)(l_att)

merged_vector = keras.layers.concatenate([l_att_contxt_do, l_att_do], axis=-1)

l_att_sent_dense = Dense(dense_num, activation=ac_func, kernel_initializer="glorot_uniform", name='final_nn')(merged_vector)
l_att_sent_drop = Dropout(dp_rate)(l_att_sent_dense)
predictions = Dense(3, activation='sigmoid', name='final_output')(l_att_sent_drop)

test_model2 = Model(inputs=[sent_input_contxt, sent_input],
        outputs=predictions)
test_model2.compile(loss='categorical_crossentropy', optimizer= 'rmsprop',
        metrics=['accuracy', 'mae'])
print(test_model2.summary())

"""Training the model"""
hist = test_model2.fit([processed_contexts, processed_dataset], code_targets,
        epochs=epoch_nums,
        batch_size=64,
        class_weight='auto')

json_string = test_model2.to_json()
with open('rnn_attention_context.json', 'w') as writefile:
    writefile.write(json_string)
# test_model2.save('webapp_ctt_ctx_att.model')
# pickle.dump(keras_tokenizer, open("keras_tokenizer.pkl", "wb"))
# pickle.dump(labels_tokenizer, open("labels_tokenizer.pkl", "wb"))