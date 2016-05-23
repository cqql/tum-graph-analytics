#!/usr/bin/env python

import argparse
import collections
import math
import os
import struct

import numpy as np

import pandas
import scipy.sparse
import sklearn.cross_validation

Split = collections.namedtuple("Split", ["prodoffset", "useroffset", "byprods",
                                         "byusers"])


def split(file, k, test_size):
    data = pandas.read_csv(file,
                           header=None,
                           names=["User", "Product", "Rating", "Timestamp"])

    # Cut off timestamp
    data = data[["User", "Product", "Rating"]]

    # Store all users and products
    users = data["User"].sort_values().unique()
    products = data["Product"].sort_values().unique()

    userkeys = {users[i]: i for i in range(users.size)}
    productkeys = {products[i]: i for i in range(products.size)}

    # Split into training and test set
    train, test = sklearn.cross_validation.train_test_split(data,
                                                            test_size=test_size)

    # Save test data
    Rtest = scipy.sparse.coo_matrix((test["Rating"],
                                     ([productkeys[p] for p in test["Product"]],
                                      [userkeys[u] for u in test["User"]])),
                                    shape=(len(products), len(users))) # yapf: disable

    # Set up sparse matrix
    R = scipy.sparse.coo_matrix((train["Rating"],
                                 ([productkeys[p] for p in train["Product"]],
                                  [userkeys[u] for u in train["User"]])),
                                shape=(len(products), len(users))) # yapf: disable

    # Convert to CSC format for slicing
    R = R.tocsc()

    # Split k-ways
    splits = []
    m, n = R.shape
    # height/width of matrix splits (if m,n are perfectly divisible by k)
    h, w = math.floor(m / k), math.floor(n / k)
    # height/width surplus that has to be split among workers
    hplus, wplus = m % k, n % k
    hoffset, woffset = 0, 0
    for i in range(k):
        hrange, wrange = h, w

        if hplus > 0:
            hrange += 1
            hplus -= 1

        if wplus > 0:
            wrange += 1
            wplus -= 1

        byprods = R[hoffset:hoffset + hrange, :]
        byusers = R[:, woffset:woffset + wrange]
        splits.append(Split(hoffset, woffset, byprods, byusers))

        hoffset += hrange
        woffset += wrange

    return users, products, splits, Rtest


def savemat(f, mat):
    f.write(struct.pack("2II", *mat.shape, mat.nnz))
    f.write(mat.row.astype(np.uint32).tobytes())
    f.write(mat.col.astype(np.uint32).tobytes())
    f.write(mat.data.astype(np.float32).tobytes())


def savesplit(path, split):
    prodoffset, useroffset, Rprods, Rusers = split

    Rprods = Rprods.tocoo()
    Rusers = Rusers.tocoo()

    with open(path, "wb") as f:
        f.write(struct.pack("I", prodoffset))
        savemat(f, Rprods)
        f.write(struct.pack("I", useroffset))
        savemat(f, Rusers)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", type=float, default=0.2, help="Test size")
    parser.add_argument("k", type=int, help="Split k-ways")
    parser.add_argument("data", help="Review data file")
    parser.add_argument("out", help="Output directory")

    args = parser.parse_args()

    if os.path.exists(args.out):
        parser.error("Output directory does already exist")

    users, products, splits, Rtest = split(args.data, args.k, args.t)

    os.makedirs(args.out)
    np.savetxt(os.path.join(args.out, "users"), users, fmt="%s")
    np.savetxt(os.path.join(args.out, "products"), products, fmt="%s")

    for i in range(args.k):
        path = os.path.join(args.out, "rank-{}".format(i))
        savesplit(path, splits[i])

    with open(os.path.join(args.out, "test"), "wb") as f:
        savemat(f, Rtest)


if __name__ == "__main__":
    main()
