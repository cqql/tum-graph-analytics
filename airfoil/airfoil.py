#!/usr/bin/env python

import argparse
import io
import sys

import numpy as np
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error


def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r",
                        "--rounds",
                        type=int,
                        default=10,
                        help="Number of boosting rounds")
    parser.add_argument("file", help="Data file")

    return parser.parse_args()


def main():
    args = parseargs()

    with open(args.file, "r") as f:
        x = np.array([[float(x) for x in line.split()] for line in f])

    y = x[:, -1]
    x = x[:, :-1]

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, train_size=0.9)

    train = xgb.DMatrix(xtrain, label=ytrain)
    test = xgb.DMatrix(xtest, label=ytest)

    params = {"silent": 1,
              "objective": "reg:linear",
              "booster": "gblinear",
              "alpha": 0,
              "lambda": 1e-6}
    watchlist = [(test, "eval"), (train, "train")]

    booster = xgb.train(params,
                        train,
                        num_boost_round=args.rounds,
                        evals=watchlist)

    ypred = booster.predict(test)

    # Mean squared error
    mse = mean_squared_error(ytest, ypred)

    print("Mean Squared Error = {:f}".format(mse))

    model = io.StringIO()
    booster.dump_model(model)
    print(model.getvalue())


if __name__ == "__main__":
    main()
