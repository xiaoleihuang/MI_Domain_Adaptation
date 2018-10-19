from gensim.models import Word2Vec
import os
import numpy as np
import glob
import h5py
from os.path import basename
import sys
import data_helper

def adaptive_retrain(fpath, sents):
    """
    Retrain the pretrained model by given dataset,
    a process of transfer learning

    fpath: input model path
    sents: list of new sentences
    """
    if fpath.endswith('txt'):
        model = Word2Vec.load(fpath, binary=False)
    else:
        model = Word2Vec.load(fpath, binary=True)

    model.build_vocab(sents)
    model.train(sents)
    return model

directory = '../MI_hdf5/*.hdf5'
filelist = glob.glob(directory)
w2v_size=200

def train_codes(code_num=10):
	code_padding = 'unknown'# for if the code_num larger than the index
	documents = []
	for filep in filelist:
		data = h5py.File(filep, 'r')
		print(filep)
		for index in range(len(data['CODE'])):
			if index < 10:
				code_list = [code_padding] * (code_num - index)
				for tmp_idx_code in range(index):
					code_list.append(data['CODE'][tmp_idx_code].decode('utf-8'))
			else:
				code_list = data['CODE'][index-10:index]
				#unicode
				code_list = [item.decode('utf-8') for item in code_list]
			code_list_new = []
			for code_tmp in code_list:
				if '+' in code_tmp or '-' in code_tmp:
					code_list_new.append(code_tmp[:-2]+str(data_helper.convert_code(code_tmp)))
				else:
					code_list_new.append(code_tmp)
			documents.append(code_list_new)
	model = Word2Vec(
		documents,min_count=1, window=5,
		size=w2v_size, iter=20, sg=0, workers=12)
	model.save('../preprocessed_data/w2v_corpus/w2v_codes.txt')

train_codes()

def train_words_corpus():
	# extract and load data from hdf5
	documents = []

	import extract_data_hdf5

	for filep in filelist:
		data = h5py.File(filep, 'r')
		print(filep)
		for index, line in enumerate(data['WORDS']):
			# append other information such as codes
			try:
				documents.append(line.decode('utf-8').strip())
			except UnicodeDecodeError:
				tmp = extract_data_hdf5.byte_utf8_converter(line.strip())
				if len(tmp) > 10:
					documents.append(tmp)

	print(len(documents))

	# for tokenize
	model = Word2Vec(
		data_helper.preproc_data(documents, split_sent=True),
		min_count=1, window=5, size=w2v_size, iter=20, sg=0, workers=12)
	model.save('../preprocessed_data/w2v_corpus/w2v_tokenize.txt')

	# for tokenize & stem
	model = Word2Vec(
		data_helper.preproc_data(documents, use_stem=True, split_sent=True),
		min_count=1, window=5, size=w2v_size, iter=20, sg=0, workers=12)
	model.save('../preprocessed_data/w2v_corpus/w2v_tokenize_stem.txt')
