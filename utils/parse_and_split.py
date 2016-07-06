import gzip
import pandas as pd
from collections import Counter
import string
import spacy
import numpy as np
import array
import itertools
from spacy import attrs

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

def getDF(path, *keys):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = {k: d[k] for k in keys}
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')  
    
def get_rev(path, index):
    for d in parse(path):
        yield unicode(d[index], 'utf-8')

def get_vocab(reviews, threshold, nlp, n_threads=4):
    counter = Counter()
    doc = nlp.pipe(reviews, n_threads, entity=False, parse=False)
    for review in doc:
        counter.update(review.count_by(spacy.attrs.LEMMA))
    return [(key,val) for key, val in counter.iteritems() 
            if val > threshold and nlp.vocab.strings[key].isalpha() and not spacy.en.English.is_stop(nlp.vocab.strings[key])]

def get_word_bags(reviews, voc, nlp):
    ret = []
    voc_pos = {word:ind for ind, word in enumerate(voc)}
    doc = nlp.pipe(reviews, n_threads=4, entity=False, parse=False)
    for review in doc:
        ret.append({voc_pos[word]:count for word, count in review.count_by(spacy.attrs.LEMMA).iteritems() \
                    if word in voc_pos})
    return ret

def from_test_to_train(train, test, criterion):
    row = test[criterion].sample(n=1, random_state=np.random.RandomState())
    train.loc[row.index.item()] = row.iloc[0]
    test.drop(row.index, inplace=True)
    
def split_train_test(df, voc, frac=0, n=0):
    train = df
    if frac != 0:
        test = train.sample(frac=frac, random_state=np.random.RandomState())
    else:
        test = train.sample(n=n, random_state=np.random.RandomState())
    train = train.drop(test.index)
    
    miss_user = test[~test['reviewerID'].isin(train['reviewerID'])]['reviewerID'].unique()
    miss_prod = test[~test['asin'].isin(train['asin'])]['asin'].unique()
    
    print "missing users products"
    print len(miss_user), len(miss_prod)
    
    print len(train), len(test)
    for mu in miss_user:
        from_test_to_train(train, test, test['reviewerID'] == mu)
    for mp in miss_prod:
        from_test_to_train(train, test, test['asin'] == mp)
    print len(train), len(test)
    miss_words = set(range(len(voc)))
    for wb in train['wordBags']:
        miss_words -= set(wb)
        if not miss_words:
            print "no missing words"
            return train, test
    for mw in miss_words:
        from_test_to_train(train, test, test['wordBags'].map(lambda wb: mw in wb))
    
    print len(train), len(test)
    return train, test

def get_k_splits(df, index, uni, k):
    length = len(uni)
    k_len = [length/k] * (k-1)+[length - length/k * (k-1)]
    k_splits = np.array(k_len).cumsum() - 1
    
    yield 0, k_len[0], df[df[index] <= uni[k_splits[0]]]
    for i in range(1,k):
        yield k_splits[i-1]+1, k_len[i], df[(uni[k_splits[i-1]] < df[index]) & (df[index] <= uni[k_splits[i]])]

def split_dict(d, k_splits):
    ret = [{key: val for key, val in d.iteritems() if (key <= k_splits[0])}]
    for i in range(1,len(k_splits)):
        ret += [{key: val for key, val in d.iteritems() if (k_splits[i-1] < key) & (key <= k_splits[i])}]
    return ret

def get_k_splits_words(df, wbdf, words, counts, k):
    #cum_counts = np.array(counts).cumsum()
    #length = cum_counts[-1]
    length = len(words)
    k_len = [length/k] * (k-1)+[length - length/k * (k-1)]
    k_splits = np.array(k_len).cumsum()
    #k_splits = np.searchsorted(cum_counts, k_splits)
    offsets = [0] + (k_splits[:len(k_splits)-1]+1).tolist()
    return offsets, df.join(wbdf.apply(lambda x: pd.Series(split_dict(x, k_splits))))

