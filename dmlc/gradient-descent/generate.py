#!/usr/bin/env python

import argparse

import scipy.stats
import numpy as np
import numpy.random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s",
                        "--sigma",
                        type=float,
                        default=1.0,
                        help="Standard deviation of Gaussian noise")
    parser.add_argument("-n", type=int, default=1000, help="Number of samples")
    parser.add_argument("-w",
                        default=[-5.5, 1.0, 2.3, 8.7],
                        type=float,
                        nargs="*",
                        help="Weight vector")
    parser.add_argument("out", help="Output path")

    args = parser.parse_args()

    w = np.asarray(args.w)
    X = numpy.random.uniform(-10, 10, (args.n, w.size))
    Y = (X @w).reshape((args.n, 1))

    # Add Gaussian noise
    Y += np.random.normal(0, args.sigma, (args.n, 1))

    data = np.concatenate((X, Y), 1)
    np.savetxt(args.out, data, fmt="%3.10f")


if __name__ == "__main__":
    main()
