"""Utilities for the review data files.
"""

import csv

import pandas
import scipy.sparse


def parse(path):
    """Parse the review data in `path` into a sparse matrix.

    Parameters
    ----------
    path : string
        Path to CSV file with review data

    Returns
    -------
    products : numpy array
        Product IDs in the order of rows of R
    users : numpy array
        User IDs in the order of columns of R
    R : numpy array
        Sparse scipy products*users matrix of ratings (COO format)
    T : numpy array
        Sparse scipy products*users matrix of rating times (COO format)
    """
    data = pandas.read_csv(path,
                           header=None,
                           names=["User", "Product", "Rating", "Timestamp"])

    # Store all users and products
    users = data["User"].sort_values().unique()
    products = data["Product"].sort_values().unique()

    # Map users and products to their keys in the sorted lists
    userkeys = {users[i]: i for i in range(users.size)}
    productkeys = {products[i]: i for i in range(products.size)}

    # Normalize timestamps
    data["Timestamp"] -= data["Timestamp"].mean()

    # Convert from seconds to days
    data["Timestamp"] /= 60 * 60 * 24

    rlocations = (data["Rating"],
                  ([productkeys[p] for p in data["Product"]],
                   [userkeys[u] for u in data["User"]])) # yapf: disable
    R = scipy.sparse.coo_matrix(rlocations, shape=(len(products), len(users)))

    tlocations = (data["Timestamp"],
                  ([productkeys[p] for p in data["Product"]],
                   [userkeys[u] for u in data["User"]])) # yapf: disable
    T = scipy.sparse.coo_matrix(tlocations, shape=(len(products), len(users)))

    return (products, users, R, T)


def generate(out, A, rows=None, cols=None):
    """Generate CSV reviews from matrix `A`.

    Parameters
    ----------
    out : IO
        IO to write the CSV to
    A : numpy array
        Matrix of ratings (can be sparse)
    rows : array, optional
        Product IDs for rows
    cols : array, optional
        User IDs for columns
    """
    writer = csv.writer(out)
    m, n = A.shape

    if rows is None:
        rows = list(range(m))

    if cols is None:
        cols = list(range(n))

    if scipy.sparse.issparse(A):
        A = A.tocoo()
        for k in range(A.nnz):
            i, j = A.row[k], A.col[k]

            writer.writerow([rows[i], cols[j], A.data[k], 0])
    else:
        for i in range(m):
            for j in range(n):
                writer.writerow([rows[i], cols[j], A[i, j], 0])
