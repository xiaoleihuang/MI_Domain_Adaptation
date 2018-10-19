import numpy as np
import os
import sys

# keras preprocessing
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras

from nltk.tokenize import TweetTokenizer
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from imblearn.over_sampling import RandomOverSampler
sampler = RandomOverSampler(random_state=0)

np.random.seed(0)  # for reproducibility

# for word2vec model
import gensim
import pickle

def load_glove(datafile):
    """
    Load pre-trained glove vector model
    """
    word2vec_model = {}
    with open(datafile, 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            try:
                coefs = np.asarray(values[1:], dtype='float32')
                word2vec_model[word] = coefs
            except ValueError:
                print(line)
    return word2vec_model


def load_word2vec(datafile, binary=True):
    """
    Load pre-trained word2vec model
    all generated word2vec models should be consistent with the following format:
        dict: key, array of scalars
    """
    if datafile.endswith('bin'):
        model = gensim.models.KeyedVectors.load_word2vec_format(datafile, binary=binary)
    else:
        model = gensim.models.KeyedVectors.load(datafile)
    w2v = dict(zip(model.wv.index2word, model.wv.syn0))
    return w2v


def split_tvt_idx(dataset, TRAIN_SPLIT=0.8, VALIDATION_SPLIT=0.1, TEST_SPLIT=0.1):
    # shuffle the idx of data
    length = len(dataset)
    shuffle_indices = np.arange(length)
    np.random.shuffle(shuffle_indices)

    train_idx = shuffle_indices[:int(TRAIN_SPLIT * length)]
    valid_idx = shuffle_indices[int(TRAIN_SPLIT * length):int((1 - TEST_SPLIT) * length)]
    test_idx = shuffle_indices[-int(TEST_SPLIT * length):]

    return train_idx, valid_idx, test_idx

def balance_oversampling(X, y):
    """
    Balance the dataset
    :param X:
    :param y:
    :return:
    """
    global sampler
    return sampler.fit_sample(X, y)

def shuffle_split_data(X, y, TRAIN_SPLIT=0.8, VALIDATION_SPLIT=0.1, TEST_SPLIT=0.1):
    # shuffle
    length = len(y)
    shuffle_indices = np.arange(length)
    np.random.shuffle(shuffle_indices)
    X = np.asarray(X)
    X = X[shuffle_indices]
    y = y[shuffle_indices]

    x_train = X[:int(TRAIN_SPLIT * length)]
    y_train = y[:int(TRAIN_SPLIT * length)]
    x_validataion = X[int(TRAIN_SPLIT * length):int((1 - TEST_SPLIT) * length)]
    y_validataion = y[int(TRAIN_SPLIT * length):int((1 - TEST_SPLIT) * length)]
    x_test = X[-int(TEST_SPLIT * length):]
    y_test = y[-int(TEST_SPLIT * length):]

    return x_train, y_train, x_validataion, y_validataion, x_test, y_test


def padding2sequences(X, MAX_NB_WORDS=1000, MAX_SEQUENCE_LENGTH=40, keras_tokenizer=None):
    """Padding sentences."""
    if keras_tokenizer is None or not isinstance(keras_tokenizer, Tokenizer):
        if keras.__version__.startswith('2'):
            tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters="")
        else:
            tokenizer = Tokenizer(nb_words=MAX_NB_WORDS, filters="")
        tokenizer.fit_on_texts(X)
    else:
        tokenizer = keras_tokenizer
    sequences = tokenizer.texts_to_sequences(X)

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    return data, tokenizer


def chars_dict_builder(X):
    """
    Create a character 2 indices mapping.
    Preprocess step for Character level CNN classification.
    """
    chars_set = set(" ".join([sent for sent in X]))
    print('total number of chars: ', len(chars_set))

    char_indices = dict((c, i) for i, c in enumerate(chars_set))
    return char_indices


def encode2char(X, maxlen, char_indices):
    """
    One hot encoding for running character-level neural networks;
    However, this method always produce too large sparse matrix
    """
    # indices_char = dict((i, c) for i, c in enumerate(chars_dict))
    char_size = len(char_indices)

    # convert X to indices
    data = np.zeros((len(X), maxlen, char_size), dtype=np.bool)

    for index, sent in enumerate(X):
        counter = 0
        sent_array = np.zeros((maxlen, char_size))
        # list(sent.lower().replace(' ', '')) to remove all whitespaces
        # need to test which works better
        chars = list(sent.lower())

        for c in chars:
            if counter >= maxlen:
                break
            else:
                char_array = np.zeros(char_size, dtype=np.int)
                if c in char_indices:
                    ix = char_indices[c]
                    char_array[ix] = 1
                sent_array[counter, :] = char_array
                counter += 1
        data[index, :, :] = sent_array
    return data


def encode2char_bme(X, maxlen, char_indices):
    """
    One hot encoding for the sentence, but another strategy:
        each word will be modeled by begin, middle and end
    """
    char_size = len(char_indices)
    data = np.zeros((len(X), maxlen * 3, char_size), dtype=np.int)

    for index, sent in enumerate(X):
        counter = 0
        sent_array = np.zeros((maxlen * 3, char_size))
        words = list(sent.lower().split())

        for w in words:
            if counter >= maxlen * 3:
                break
            else:
                begin_array = np.zeros(char_size, dtype=np.int)
                if w[0] in char_indices:
                    ix = char_indices[w[0]]
                    begin_array[ix] = 1
                sent_array[counter, :] = begin_array
                counter += 1

                mid_array = np.zeros(char_size, dtype=np.int)
                if len(w) > 2:
                    for c in w[1:-1]:
                        if c in char_indices:
                            ix = char_indices[c]
                            mid_array[ix] = 1
                sent_array[counter, :] = mid_array
                counter += 1

                end_array = np.zeros(char_size, dtype=np.int)
                if len(w) > 1:
                    if w[-1] in char_indices:
                        ix = char_indices[w[-1]]
                        end_array[ix] = 1
                sent_array[counter, :] = end_array
                counter += 1
        data[index, :, :] = sent_array
    return data


def mini_batch_generator(X, y, batch_size=128):
    """
    Mini batch data generator
    """
    for i in range(0, len(X), batch_size):
        x_sample = X[i: i + batch_size]
        y_sample = y[i: i + batch_size]
        yield (x_sample, y_sample)


# define stemmer and tokenizer
tokenizer_twitter = TweetTokenizer()
stemmer = SnowballStemmer("english")


def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def mytokenizer(text):
    tokens = tokenizer_twitter.tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems


class BaseConfiguration:
    def __init__(self):
        self.learning_rate = 0.01
        self.training_iters = 40000
        self.vocabulary_size = 10000
        self.batch_size = 50
        self.display_step = 1
        self.val_step = 50
        self.utterances = 10
        # Network Parameters
        self.seq_max_len = 50  # Sequence max length
        self.seq_min_len = 5
        self.embedding_size = 200
        self.n_hidden = 100  # hidden layer num of features
        self.n_classes = 2  # linear sequence or not
        # self.num_layers = 1
        # self.keep_prob = 1
        # self.debug_sentences = True
        self.path = os.getcwd()
        self.filenames = filter(lambda x: x[-1] == '5', os.listdir('./'))

    def printConfiguration(self):
        # print configuration
        print('---------- Configuration: ----------')
        print('learning_rate', self.learning_rate)
        print('training_iters', self.training_iters)
        print('batch_size', self.batch_size)
        print('vocab_size', self.vocabulary_size)
        # Network Parameters
        print('seq_max_len', self.seq_max_len)  # Sequence max length
        print('seq_min_len', self.seq_min_len)
        print('embedding_size', self.embedding_size)
        print('n_hidden', self.n_hidden)  # hidden layer num of features
        print('n_classes', self.n_classes)  # linear sequence or not
        print('utterances in a sequence', self.utterances)
        # print('dataset size', self.file_len)
        # print('num_layers', self.num_layers)
        # print('keep_prob (dropout = 1-keep_prob)', self.keep_prob)
        print('------------------------------------')


def cal_dist(weights, codes, keys_list=[-1, 0, 1]):
    """ Compute Distribution of CODES and generate results for PyPlot

    Args:
        weights (list):
        codes (list):
        keys_list (list): optional

    Returns:
        dist_dict (dict): a list of weights for each key
    """
    dist_dict = dict.fromkeys(keys_list, list())

    for weight, code in zip(weights, codes):
        if '+' in code or '-' in code:
            if int(code[-2:]) > 0:
                dist_dict[1].append(weight)
            else:
                dist_dict[-1].append(weight)
        else:
            dist_dict[0].append(weight)

    return dist_dict


def convert_code(code):
    if '+' in code or '-' in code:
        if int(code[-2:]) > 0:
            return 1
        else:
            return -1
    else:
        return 0


def encode_codes(codes):
    """Encode CODES to +1, 0, -1, such as 'O+2' and 'O+3' will be +1;
    'O-3' and 'O-4' will be -1.

    Examples:

    Args:
        codes (list): list of codes

    Return:
        list of encoded codes
    """
    return [convert_code(item) for item in codes]


def cal_embedding(sents, model_path, word_idx, mode='mean'):
    """Calculate the embedding for each sentence

    Args:
        sents (list): a list of sentences
        model_path (str): keras model path
        word_idx (dict): a dictionary of mapping between words and indices
        mode (str): current support mean and concatenate word vectors
    """
    # check the mode
    if mode not in ['mean', 'concatenate']:
        print('Only mean and concatenate are allowed currently!')
        sys.exit(0)

    # get weights of embedding layer from the model
    from keras.models import load_model
    k_model = load_model(model_path)
    embd_weights = k_model.layers[0].get_weights()[0]

    # loop through the sentences to calculate embedding for each sentence
    for sent in sents:
        sent = sent.strip()
        if len(sent) < 1:  # control the length of sentences
            continue

        # simple tokenization by whitespace
        idx_list = [word_idx.get(word.strip().lower(), -1)
                    for word in sent.split(' ') if len(word.strip()) > 0]

        # calculate embeddings
        sent_embedding = []
        if mode == 'mean':
            for tmp_idx in idx_list:
                if tmp_idx == -1 or tmp_idx > len(embd_weights) - 1:
                    sent_embedding.append(list(np.zeros(len(embd_weights[0]))))
                else:
                    sent_embedding.append(embd_weights[tmp_idx])
            sent_embedding = np.mean(sent_embedding, axis=0)
        elif mode == 'concatenate':
            for tmp_idx in idx_list:
                if tmp_idx == -1 or tmp_idx > len(embd_weights) - 1:
                    sent_embedding.extend(list(np.zeros(len(embd_weights[0]))))
                else:
                    sent_embedding.extend(embd_weights[tmp_idx])
        yield sent_embedding


def preproc_data(data, use_lower=True, use_stem=False, use_stopwords=False, split_sent=False):
    """Functions to preprocess the dataset"""
    assert type(data) != str, "data must be a list or array"
    dataset = []

    for doc in data:
        if use_lower:
            doc = doc.lower()
        doc = doc.strip()
        doc = ''.join([wchar if wchar.isalnum() or wchar == '\'' else ' ' for wchar in doc])

        tmp_doc = word_tokenize(doc)
        tmp_doc = [word.strip() for word in tmp_doc if len(word.strip()) > 0]
        if use_stem:
            tmp_doc = [stemmer.stem(word) for word in tmp_doc]
        if use_stopwords:
            stopwords_set = set(stopwords.words('english'))
            tmp_doc = [word for word in tmp_doc if word not in stopwords_set]
        if not split_sent:
            tmp_doc = " ".join(tmp_doc)
        dataset.append(tmp_doc)
    return dataset

def add_extra_docs(fpath):
    """

    :param fpath: The file path of the extra training document
    :return:
    """
    contxt = []
    content = []
    labels = []
    if sys.version_info[0] == 2:
        reload(sys)
        sys.setdefaultencoding('utf8')

    with open(fpath) as datafile:
        datafile.readline() # skip the 1st line, column names
        for line in datafile:
            line = line.strip()
            if len(line) < 3:
                continue
            infos = line.split('\t')
            contxt.append(infos[0].strip())
            content.append(infos[1].strip())
            labels.append(infos[2].strip())



    return content, contxt, labels

# keras2gensimdic('../preprocessed_data/keras_tokenizer.pkl')
