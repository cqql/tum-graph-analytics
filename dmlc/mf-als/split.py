#!/usr/bin/env python

import argparse
import os
import struct

import numpy as np

import pandas
import scipy.sparse
import sklearn.cross_validation


def split(file, k):
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
                                                            test_size=0.2)

    # Set up sparse matrix
    R = scipy.sparse.coo_matrix((train["Rating"],
                                 ([productkeys[p] for p in train["Product"]],
                                  [userkeys[u] for u in train["User"]]))) # yapf: disable

    # Convert to CSC format for slicing
    R = R.tocsc()

    # Split k-ways
    splits = {}
    m, n = R.shape
    h, w = float(m) / k, float(n) / k
    for i in range(k):
        byprods = R[int(i * h):int(min((i + 1) * h, m)), :]
        byusers = R[:, int(i * w):int(min((i + 1) * w, n))]
        splits[i] = (byprods, byusers)

    return users, products, splits


def savemat(path, mat):
    with open(path, "wb") as f:
        f.write(struct.pack("2II", *mat.shape, mat.nnz))
        f.write(mat.row.astype(np.uint32).tobytes())
        f.write(mat.col.astype(np.uint32).tobytes())
        f.write(mat.data.astype(np.float32).tobytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("k", type=int, help="Split k-ways")
    parser.add_argument("data", help="Review data file")
    parser.add_argument("out", help="Output directory")

    args = parser.parse_args()

    if os.path.exists(args.out):
        parser.error("Output directory does already exist")

    users, products, splits = split(args.data, args.k)

    os.makedirs(args.out)
    np.savetxt(os.path.join(args.out, "users"), users, fmt="%s")
    np.savetxt(os.path.join(args.out, "products"), products, fmt="%s")

    for i in range(args.k):
        Rprods, Rusers = splits[i]

        Rprods = Rprods.tocoo()
        Rusers = Rusers.tocoo()

        savemat(os.path.join(args.out, "rank-{}-prods".format(i)), Rprods)
        savemat(os.path.join(args.out, "rank-{}-users".format(i)), Rusers)


if __name__ == "__main__":
    main()
