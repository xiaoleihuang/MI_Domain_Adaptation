#Michelle Morales
#Extract LIWC features using 2015 English version
from collections import defaultdict
import re

def liwc(words, liwc_file = 'LIWC2015_English.dic'):#words is a list of words
    """ Get features using LIWC 2015. categories in total."""
    categoryIDs = {} #keep track of each category number
    liwcD = {} #create liwc dictionary where each liwc dictionary word is a key that maps to a list that contains the liwc categories for that word
    #path to LIWC dict
    read = open(liwc_file,'r').readlines()
    header = read[1:77] #change this number depending on how many liwc categories you want to use
    for line in header:
        items = line.strip().split()
        number,category_name = items[0],items[1]
        categoryIDs[number]=category_name

    liwc_words = read[88:]#liwc dictionary words
    for line in liwc_words:
        items = line.strip().split('\t')
        word = items[0].replace('(','').replace(')','')
        word_cats = items[1:]
        liwcD[word] = word_cats
    total_words = len(words)
    line = ' '.join(words)
    feats = defaultdict(int)#keep track of liwc frequencies
    for word in sorted(liwcD.keys()): #first 9 words are emojis with special characters TODO: treat them separately
        cats = liwcD[word] #list of liwc categories
        if '*' in word:
            pattern = re.compile(' %s'%word.replace('*',''))
        else:
            pattern = re.compile(' %s '%word)
        matches = [(m.start(0), m.end(0)) for m in re.finditer(pattern, line)] #check is liwc word is in sentence
        if matches != []: #count matches
            for C in cats:
                feats[int(C)]+=len(matches)
        else:
            for C in cats:
                feats[int(C)] += 0
    if total_words != 0: #if 0 zero words in sentence - create zero vector
        liwc_features = [(float(feats[key])/total_words) for key in sorted(feats)]
    else:
        liwc_features = ','.join([0]*73)
    category_names = [categoryIDs[str(c)] for c in sorted(feats)]
    return category_names, liwc_features
