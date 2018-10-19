"""This script is to test whether integrating lda works or not
Thus, this model is to start with the simplest model.
"""
import numpy as np
import pickle
from collections import Counter
from gensim.corpora import Dictionary
from utils import lda_helper, data_helper, model_helper

import keras
from keras.layers import Input, Dense, Dropout, GRU, Bidirectional
from keras.models import Model

from sklearn.metrics import classification_report

# parameters for initial configuration
config = data_helper.BaseConfiguration()
config.MAX_SENTS = 10

print('preparing dataset')
"""preprare dataset """
dataset_path = './preprocessed_data/dataset.pkl'
data_whole = pickle.load(open(dataset_path, 'rb'))

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
#ctt_lda_idx = lda_helper.doc2idx(lda_tokenizer, dataset, config.seq_max_len)
## convert doc_raw to bags of words
#doc_topics = lda_helper.doc2topics(lda_tokenizer, lda_model, data_whole['dataset_raw'])

ctt_lda_idx = data_whole['ctt_lda_idx']

# get training data
x_train = processed_dataset[train_indices]
x_valid = processed_dataset[valid_indices]
x_test = processed_dataset[test_indices]

ctt_lda_train = ctt_lda_idx[train_indices]
ctt_lda_valid = ctt_lda_idx[valid_indices]
ctt_lda_test = ctt_lda_idx[test_indices]

doc_topics_train = doc_topics[train_indices]
doc_topics_valid = doc_topics[valid_indices]
doc_topics_test = doc_topics[test_indices]

# get labels
code_train = code_targets[train_indices]
code_valid = code_targets[valid_indices]
code_test = code_targets[test_indices]

# load pretrained embeddings
w2v_path = './preprocessed_data/w2v_corpus/glove.42B.300d.txt'
if 'glove' in w2v_path:
    w2v_model = data_helper.load_glove(w2v_path)
else:
    w2v_model = data_helper.load_word2vec(w2v_path)

print('Train Length: ' + str(len(train_indices)))
print('Valid Length: ' + str(len(valid_indices)))
print('Test Length: ' + str(len(test_indices)))

print('preparing the model')
"""Prepare model"""
config.embedding_size = len(w2v_model['the'])
# initialize weights for embeddings
ctt_weights = model_helper.init_weights(w2v_model, keras_tokenizer, config.embedding_size)
ctt_embedding = model_helper.build_embedding(ctt_weights, config.seq_max_len, name='ctt_embed')

lda_weights = lda_helper.init_weight('./preprocessed_data/lda/lda.model', 20)
lda_word_embed = lda_helper.build_embedding(lda_weights, config.seq_max_len, name='lda_embed')

# define layers for ctt
ctt_input = Input(shape=(config.seq_max_len,), dtype='int32', name='ctt_input')
ctt_embed = ctt_embedding(ctt_input)
ctt_lstm = Bidirectional(GRU(100, recurrent_dropout=0.1, kernel_initializer="glorot_uniform", recurrent_activation='tanh'),
                                name='ctt_bilstm')(ctt_embed)
ctt_dropout = Dropout(0.2)(ctt_lstm)

# define layers for topic-word
lda_word_input = Input(shape=(config.seq_max_len,), dtype='int32', name='ctt_input')
lda_embed = lda_word_embed(lda_word_input)
lda_lstm = Bidirectional(GRU(20, recurrent_dropout=0.1, kernel_initializer="glorot_uniform", recurrent_activation='tanh'),
                                name='lda_word_bilstm')(ctt_embed)
lda_dropout = Dropout(0.2)(lda_lstm)


merged_vector = keras.layers.concatenate([ctt_dropout, lda_dropout], axis=-1)
merged_dense = Dense(100, activation='relu',
                     kernel_initializer="glorot_uniform", name='final_nn')(merged_vector)
last_drop = Dropout(0.1)(merged_dense)
predictions = Dense(3, activation='sigmoid', name='final_output')(last_drop)

test_model = Model(inputs=[ctt_input, lda_word_input], outputs=predictions)
# official document shows RMSprop is a better choice for recurrent neural network
test_model.compile(loss='categorical_crossentropy', optimizer= 'rmsprop',
            metrics=['accuracy'])
print(test_model.summary())


hist = test_model.fit([x_train, ctt_lda_train], code_train,
            epochs=1,
            batch_size=64,
            validation_data=([x_valid, ctt_lda_valid], code_valid),
            class_weight='auto')


y_pred = test_model.predict([x_test, ctt_lda_test]);print()
report = classification_report([np.argmax(item) for item in y_pred], [np.argmax(item) for item in code_test])
print(report)
