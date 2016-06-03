#!/usr/bin/env python

import argparse
import math


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=int, help="Split n ways")
    parser.add_argument("file", help="File to split")

    args = parser.parse_args()

    with open(args.file, "r") as f:
        lines = f.readlines()

    N = int(math.ceil(float(len(lines)) / args.n))

    for i in range(args.n):
        with open("{}-{}".format(args.file, i), "w") as f:
            f.writelines(lines[i * N:(i + 1) * N])


if __name__ == "__main__":
    main()
