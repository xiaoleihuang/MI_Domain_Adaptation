"""
This script provides basic functions for topic model
"""
import numpy as np

from gensim.corpora import Dictionary
from gensim.models import LdaModel

from keras.layers import Embedding
import os
import pickle


def load_lda(lda_path):
    return LdaModel.load(lda_path)


def keras2gensimdic(kt_path):
    """
    The function convert keras's tokenizer to gensim.corpora.Dicionary type.
    :param kt_path:
    :return: an object of gensim.corpora.Dicionary
    """
    kt = pickle.load(open(kt_path, 'rb'))

    tmp_path = './tmp.txt'
    with open(tmp_path, 'w') as writefile:
        for word in sorted(kt.word_index.keys()):
            writefile.write(str(kt.word_index[word])+'\t'
                            + word + '\t'
                            + str(kt.word_docs[word])+'\n')

    from gensim.corpora.dictionary import Dictionary
    gdicts = Dictionary.load_from_text(tmp_path)
    os.remove(tmp_path)
    return gdicts


def docs2dicts_idx(docs):
    """
    Convert input raw documents into gensim's dictionary
    :param docs:
    :return:
    """
    # check the type of each document
    # , because the gensim only accept list of words
    docs_clean = [doc.split(' ') for doc in docs]
    dictionary = Dictionary(docs_clean)

    doc_matrix = [dictionary.doc2bow(doc) for doc in docs_clean]
    dictionary.save('lda_dict.pkl')
    return dictionary, doc_matrix


def doc2idx(dictionary, docs, max_len):
    """
    Convert docs into lists of idx.
    :param dictionary:
    :param docs:
    :return:
    """
    idx_clean = np.zeros(shape=[len(docs), max_len])
    for idx_doc, doc in enumerate(docs):
        for idx_word, word in enumerate(doc.split()):
            if idx_word >= max_len:
                continue
            if word in dictionary.token2id:
                idx_clean[idx_doc, idx_word] = dictionary.token2id[word]
    return idx_clean


def train_lda(doc_matrix, dictionary, ktopic=20, alpha='symmetric', eta=None):
    """
    Train a topic model
    :param doc_matrix:
    :param dictionary:
    :param ktopic:
    :param alpha:
    :param eta:
    :return:
    """
    ldamodel = LdaModel(doc_matrix,
            id2word=dictionary, num_topics=ktopic,
            passes=2, alpha=alpha, eta=None)
    return ldamodel


def init_weight(lda_path, embedding_size):
    """
    Initialize embedding's weights by LDA
    :param lda_path:
    :param embedding_size:
    :return:
    """
    # load the model
    ldamodel = LdaModel.load(lda_path)
    # initialize weight matrix
    embedding_matrix = np.zeros((ldamodel.num_terms + 1, ldamodel.num_topics))

    # assign value to the matrix
    for idx, word in ldamodel.id2word.items():
        embedding_vector = np.zeros(embedding_size)
        for tmp_pair in ldamodel.get_term_topics(idx, minimum_probability=-0.1):
            embedding_vector[tmp_pair[0]] = tmp_pair[1]
        # if len(embedding_vector) != embedding_size:
        #     if len(embedding_vector) >= embedding_size:
        #         embedding_vector = embedding_vector[:embedding_size]
        #     else:
        #         # pre padding with zeros
        #         embedding_vector = np.zeros(embedding_size - len(embedding_vector)+1) + embedding_vector
        if embedding_vector is not None:
            embedding_matrix[idx] = embedding_vector
    return embedding_matrix


def build_embedding(embedding_matrix, max_len, name):
    """
    Build embedding by lda
    :param max_len:
    :param name:
    :return:
    """
    # build embedding with initial weights
    topic_emmd = Embedding(embedding_matrix.shape[0],
                       embedding_matrix.shape[1],
                       weights=[embedding_matrix],
                       input_length=max_len,
                       trainable=True,
                       name=name)
    return topic_emmd


def doc2topics(dictionary, ldamodel, docs_raw):
    """
    Convert document into topic distributions
    :param dictionary:
    :param ldamodel:
    :param docs_raw:
    :return:
    """
    doc_topic_matrix = [
        [
            item[1] for item in ldamodel.get_document_topics(dictionary.doc2bow(doc.split()), minimum_probability=0.0)
        ] for doc in docs_raw
    ]
    doc_topic_matrix = np.asarray(doc_topic_matrix)
    return doc_topic_matrix


def extract_eta(neural_model, layer_name):
    """
    Extract topic*num_word matrix, which will then be used for initialize LDA
    :param neural_model:
    :param layer_name:
    :return:
    """
    eta = np.transpose(neural_model.get_layer(layer_name).get_weights()[0])
    return eta

if __name__ == '__main__':
    # import pickle
    # data_whole = pickle.load(open('../preprocessed_data/dataset.pkl', 'rb'))
    # # in order to use the same index of keras_tokenizer
    # # dictionary = keras2gensimdic('../preprocessed_data/keras_tokenizer.pkl')
    # # dictionary.save('lda_dict.pkl')
    # # doc_matrix = doc_matrix = [dictionary.doc2bow(doc.split()) for doc in data_whole['dataset_raw'] + data_whole['context_raw']]
    # dictionary, doc_matrix = docs2dicts_idx(data_whole['dataset_raw']+data_whole['context_raw'])
    #
    # ldamodel = train_lda(doc_matrix, dictionary)
    # ldamodel.save('lda.model')
    print(init_weight('../preprocessed_data/lda/lda.model', 20))