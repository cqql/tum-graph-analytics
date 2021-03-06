import pandas as pd
import os
import string
import spacy
import numpy as np
import array
import itertools
import json
import argparse
import time
from parse_and_split import *

def save(f, offset, rows, cols, bags, words, vals, n_rows=0, n_cols=0, n_words=0):
    assert len(rows) == len(cols) == len(bags), "Wrong list length for save"
    if n_rows == 0:
        n_rows = max(rows)+1
    if n_cols == 0:
        n_cols = max(cols)+1
    if n_words == 0:
        n_words = max(words)+1
    array.array("I", [offset, n_rows, n_cols, n_words, len(rows), len(vals)]).write(f)
    array.array("I", rows).write(f)
    array.array("I", cols).write(f)
    array.array("I", bags).write(f)
    array.array("I", words).write(f)
    array.array("f", vals).write(f)


def save_df(df, path, userkeys, prodkeys, offset, n_rows=0, n_cols=0, n_words=0):
    sorteddf = df.sort_values(['reviewerID', 'asin'])
    users = [userkeys[u] for u in sorteddf['reviewerID']]
    prods = [prodkeys[p] for p in sorteddf['asin']]
    bags = np.array([len(wb) for wb in sorteddf['wordBags']]).cumsum() # maybe -1?
    wbtmp = [sorted([(pos, count) for pos, count in wb.iteritems()], key=lambda x: x[0]) for wb in sorteddf['wordBags']]
    words, values = zip(*list(itertools.chain.from_iterable(wbtmp)))

    with open(path, 'wb') as f:
        save(f, offset, users, prods, bags, words, values, n_rows, n_cols, n_words)


def save_df_word(df, path, userkeys, prodkeys, offset, k):
    sorteddf = df.sort_values(['reviewerID', 'asin'])
    users = [userkeys[u] for u in sorteddf['reviewerID']]
    prods = [prodkeys[p] for p in sorteddf['asin']]
    for ind in range(k):
        bags = np.array([len(wb) for wb in sorteddf[ind]]).cumsum() # maybe -1?
        wbtmp = [sorted([(pos - offset[ind], count) for pos, count in wb.iteritems()], key=lambda x: x[0]) for wb in sorteddf[ind]]
        words, values = zip(*list(itertools.chain.from_iterable(wbtmp)))
        with open(path + `ind`, 'wb') as f:
            save(f, offset[ind], users, prods, bags, words, values, n_words=max(words)+1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", type=float, default=0.25, help="Fraction of test set")
    parser.add_argument("-v", type=float, default=0.33, help="Fraction of validation set of total training set") 
    parser.add_argument("-w", type=int, default=10, help="Threshold of lowest frequency for a word to be considered for the vocabulary.")
    parser.add_argument("k", type=int, help="Split k-ways")
    parser.add_argument("data", help="Review data file")
    parser.add_argument("out", help="Output directory")

    args = parser.parse_args()
    
    # get eng
    start = time.clock()
    eng = spacy.en.English()
    end = time.clock()
    print "eng", end - start

    # build vocab
    start = time.clock()
    voc, count = zip(*get_vocab(get_rev(args.data, 'reviewText'), args.w, eng))
    end = time.clock()
    print "build vocab done in ", end - start

    # get word frequencies per review
    start = time.clock()
    bags = get_word_bags(get_rev(args.data, 'reviewText'), voc, eng)
    end = time.clock()
    print "word bags done in ", end - start
    
    # parse dataframe
    start = time.clock()
    df = getDF(args.data, 'reviewerID', 'asin')
    end = time.clock()
    print "got user/prod in ", end - start

    # unique lists
    users = df['reviewerID'].sort_values().unique()
    prods = df['asin'].sort_values().unique()
    userkeys = {u: i for i, u in enumerate(users)}
    prodkeys = {p: i for i, p in enumerate(prods)}
    
    # put bags into pd df
    wbdf = pd.DataFrame({'wordBags':bags})

    # df used to split in user and prod dimensions
    df_total = pd.concat([df,wbdf], axis=1, join='inner', copy='false')
    df_train, df_test = split_df(df_total, voc, frac=args.t)
    df_train, df_validation = split_df(df_train, voc, frac=args.v)
    

    # save user-split train files
    for ind, (offset, size, df_user) in enumerate(get_k_splits(df_train, 'reviewerID', users, args.k)):
        outpath = os.path.join(args.out, "_user_train" + `ind`)
        save_df(df_user, outpath, userkeys, prodkeys, offset, n_rows=size, n_words=len(voc))
    print "saved usersplit train"
    # save product-split train files
    for ind, (offset, size, df_prod) in enumerate(get_k_splits(df_train, 'asin', prods, args.k)):
        outpath = os.path.join(args.out, "_prod_train" + `ind`)
        save_df(df_prod, outpath, userkeys, prodkeys, offset, n_cols=size, n_words=len(voc))
    print "saved prodsplit train"
    # save words-split train files
    offset, df_word = get_k_splits_words(df_train, df_train['wordBags'], voc, count, args.k)

    outpath = os.path.join(args.out, "_word_train")
    save_df_word(df_word, outpath, userkeys, prodkeys, offset, args.k)
    print "saved wordsplit train"
    
    # save validation set split along the users
    for ind, (offset, size, df_v) in enumerate(get_k_splits(df_validation, 'reviewerID', users, args.k)):
        outpath = os.path.join(args.out, "_validation" + `ind`)
        save_df(df_v, outpath, userkeys, prodkeys, offset, n_rows=size, n_words=len(voc))
    print "saved validation split"
    # save test set split along the users
    for ind, (offset, size, df_t) in enumerate(get_k_splits(df_test, 'reviewerID', users, args.k)):
        outpath = os.path.join(args.out, "_test" + `ind`)
        save_df(df_t, outpath, userkeys, prodkeys, offset, n_rows=size, n_words=len(voc))
    print "saved test split"
    
    # document some meta data
    json.dump([{'users': len(users), 'products': len(prods), 'words' : len(voc), 'train' : len(df_train), 'test' : len(df_test), 'validation' : len(df_validation)}, 
            {'vocab' : [eng.vocab.strings[w] for w in voc]}], 
            open(os.path.join(args.out, "meta.txt"), "w"))

    # write config file necessary for running the tf program
    run_config = "\n".join(["clients=1", "id=0", "workers="+str(args.k), "iterations=30", "users="+str(len(users)), "products="+str(len(prods)), "words="+str(len(voc)), 
                        "seed=0", "atol=0.01", "rtol=0.01", "lambda=0.5", "clamp=true", "reg=false", "reg-thr=1", "stop-tol=2"])
    with open(os.path.join(args.out, "run_config.cfg"), "w") as f:
        f.write(run_config)

if __name__ == "__main__":
    main()
