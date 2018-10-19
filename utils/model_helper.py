"""
Part of the codes come from 
1. https://github.com/philipperemy/keras-attention-mechanism
2. https://github.com/richliao/textClassifier/blob/master/textClassifierHATT.py
For the documentation, refer to https://github.com/philipperemy/keras-attention-mechanism#application-of-attention-at-input-level
"""
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division

"""Keras APIs"""
import keras.backend as K
from keras.layers.recurrent import Layer
import keras
# to prevent version conflicts
if keras.__version__.startswith('1'):
    import keras.initializations as initializers
else:
    import keras.initializers as initializers
import keras.constraints as constraints
import keras.regularizers as regularizers
from keras.layers.embeddings import Embedding

import numpy as np

if 'theano' not in keras.backend.backend().lower():
    import warnings

    warnings.warn(
        'If you are using Attention model, plz use Theano as the backend of keras. Because the dot function of Tensorflow backend does not work.')


def get_activations(model, layer_idx, x_batch):
    activate_func = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer_idx].output, ])
    activations = activate_func([x_batch, 0])  # 0 is the test mode
    return activations


class Attention(Layer):
    """ Attention Layer from the paper: https://arxiv.org/abs/1409.0473
    Part of code comes from: https://groups.google.com/forum/#!topic/keras-users/suKYo6L1bSI
    The code supports adding context vectors to the attention weights; for example, the context vector could be weight vectors for each wordor weight vectors for each sentence.

    Agrs:
        use_context (bool): a boolean value to decide whether to use context vectors for each word or sentece
        context_vectors (array): an array vectors of context, if the input shape is n*m, the context should be m*p
    """

    def __init__(self,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='one',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 use_context=False,
                 context_vectors=None, **kwargs):

        self.name = 'Attention'
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.supports_masking = True
        # context settings
        self.use_context = use_context
        self.context_vectors = context_vectors
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.kernel = self.add_weight(shape=(input_shape[-1],),
                                      initializer=self.kernel_initializer,
                                      name='{}_kernel_Weights'.format(self.name),
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(input_shape[1],),
                                        initializer=self.kernel_initializer,
                                        name='{}_bias'.format(self.name),
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.built = True

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, x, mask=None):
        eij = K.dot(x, self.kernel)
        if self.use_bias:
            eij += self.bias
        if self.use_context and self.context_vectors:
            eij = K.dot(eij, self.context_vectors)
        eij = K.tanh(eij)

        # compute attention weights
        a = K.exp(eij)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        """in some cases especially in the early stages of training the sum may be almost zero; and this results in NaN's. A workaround is to add a very small positive number epslon to the sum."""
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def get_ouput_shape_for(self, input_shape):
        """For keras 1.x """
        return input_shape[0], input_shape[-1]

    def compute_output_shape(self, input_shape):
        """For keras 2.x"""
        return (input_shape[0], input_shape[-1])


def extract_layer_weights(keras_model, name):
    """
    Extract specific layer of the model by name
    :param keras_model:
    :param name:
    :return:
    """
    tmp_layer = None
    try:
        tmp_layer = keras_model.get_layer(name)
    except Exception as e:
        print(e)

    return tmp_layer.get_weights()[0]


def init_weights(w2v_model, keras_tokenizer, embedding_size=100):
    """
    Initialize a weight matrix
    :param w2v_model:
    :param keras_tokenizer:
    :param embedding_size:
    :return:
    """
    embedding_matrix = np.zeros((len(keras_tokenizer.word_index) + 1, embedding_size))

    for word, i in keras_tokenizer.word_index.items():
        embeddings_vector = w2v_model.get(word)
        # if word is found in the model, will be zeros
        if embeddings_vector is not None:
            embedding_matrix[i] = embeddings_vector
    return embedding_matrix


def build_embedding(embedding_matrix, max_len, name):
    """
    Build a Embedding layer with pre-trained weights.
    :param embedding_matrix:
    :param max_len:
    :param name:
    :return:
    """
    return Embedding(embedding_matrix.shape[0],
              embedding_matrix.shape[1],
              weights=[embedding_matrix],
              input_length=max_len,
              trainable=True,
              name=name)