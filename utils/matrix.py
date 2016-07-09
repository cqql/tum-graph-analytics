import struct

import numpy as np

import scipy.sparse


def save(f, A):
    f.write(struct.pack("2II", A.shape[0], A.shape[1], A.nnz))
    f.write(A.row.astype(np.uint32).tobytes())
    f.write(A.col.astype(np.uint32).tobytes())
    f.write(A.data.astype(np.float32).tobytes())


def load(f):
    m, n, nnz = struct.unpack("3I", f.read(struct.calcsize("3I")))
    row = np.frombuffer(f.read(nnz * struct.calcsize("I")), dtype=np.uint32)
    col = np.frombuffer(f.read(nnz * struct.calcsize("I")), dtype=np.uint32)
    val = np.frombuffer(f.read(nnz * struct.calcsize("f")), dtype=np.float32)

    return scipy.sparse.coo_matrix((val, (row, col)), shape=(m, n))


def loadfile(path):
    with open(path, "rb") as f:
        return load(f)


def savesplit(path, offset, R):
    with open(path, "wb") as f:
        f.write(struct.pack("I", offset))
        save(f, R)


def loadsplit(path):
    with open(path, "rb") as f:
        offset = struct.unpack("I", f.read(struct.calcsize("I")))
        A = load(f)

    return offset[0], A
