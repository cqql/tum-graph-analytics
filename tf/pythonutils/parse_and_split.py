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

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')    
    
def get_rev(path, index):
    for d in parse(path):
        yield unicode(d[index], 'utf-8')

def exclude(word):
    if word.isdigit() or word.isspace() or word in string.punctuation:
        return True
    else:
        return False

def get_frequent_words(reviews, threshold, nlp, n_threads=4):
    counter = Counter()
    doc = nlp.pipe(reviews, n_threads, entity=False, parse=False)
    for review in doc:
        words = review.to_array([spacy.attrs.LEMMA, spacy.attrs.IS_ALPHA, spacy.attrs.IS_STOP])
        for w in words[(words[:,1] == 1) & (words[:,2] == 0)][:,0]:
            counter[w] += 1
    return [(key,val) for key, val in counter.iteritems() if val > threshold]

def get_word_bags(voc, reviews, nlp):
    ret = []
    voc_pos = {word:ind for ind, word in enumerate(voc)}
    doc = nlp.pipe(reviews, n_threads=4, entity=False, parse=False)
    for review in doc:
        #yield review.count_by(spacy.attrs.LEMMA).keys(), review.count_by(spacy.attrs.LEMMA).values()
        ret.append({voc_pos[word]:count for word, count in review.count_by(spacy.attrs.LEMMA).iteritems() \
                    if word in voc_pos})
    return ret
    
def split_train_test(df, frac=0, n=0):
    train = df
    if frac != 0:
        test = train.sample(frac=frac, random_state=np.random.RandomState())
    else:
        test = train.sample(n=n, random_state=np.random.RandomState())
    train = train.drop(test.index)
    
    miss_user = [user for user in test['reviewerID'] if user not in train['reviewerID'].values]
    miss_prod = [prod for prod in test['asin'] if prod not in train['asin'].values]
    
    put_back = test[test['reviewerID'].isin(miss_user) & test['asin'].isin(miss_prod)]
    print len(put_back)
    
    miss_user = [user for user in miss_user if user not in put_back['reviewerID'].values]
    miss_prod = [prod for prod in miss_prod if prod not in put_back['asin'].values]
    
    print len(miss_user), len(miss_prod)
    print len(train), len(test)
    for mu in miss_user:
        tmp = test[test['reviewerID'] == mu].sample(n=1, random_state=np.random.RandomState())
        put_back.append(tmp)
    for mp in miss_prod:
        tmp = test[test['asin'] == mp].sample(n=1, random_state=np.random.RandomState())
        put_back.append(tmp)
    
    return pd.concat([train, put_back]), test.drop(put_back.index)

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

