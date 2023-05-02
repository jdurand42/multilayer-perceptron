import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score, accuracy_score
from sklearn.metrics import confusion_matrix

from utils import *
from MLP import MultiLayerPerceptron
import pickle

import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--data_path', type=str, default='../data/data.csv',
                    help='path to csv file containing data')
parser.add_argument('--model_path', type=str, default="mlp.pkl")
parser.add_argument('--raw', action="store_true")
parser.add_argument('--no_print', action="store_false")
parser.add_argument('--head', type=int, default=15)

if __name__ == "__main__":
    args = parser.parse_args()

    df = get_data(args.data_path, headers=["id", "diagnosis"])
    X, Y = get_X_Y(df, labels=["diagnosis"], drops=['id'])
    pred = Y.copy()
    Y = labelize_Y(Y, y_label="diagnosis", value="M")

    raw = X.copy()

    print(df.describe())

    print(X.head(2))
    print(Y.head(2))

    mlp = pickle.load(open(args.model_path, "rb"))

    stds = mlp.normalization['stds']
    means = mlp.normalization['means']

    X = zscore(X, stds, means)

    print(X.head(5))

    y_pred = mlp.predict(X.to_numpy(), raw=args.raw)
    pred['diagnosis'] = y_pred
    print(pred.head(args.head))
    # df = pd.DataFrame(y_pred, names=["diagnosis"])
    if args.raw is False:
        pred = unlabelize_Y(pred, y_label="diagnosis", values=("B", "M"))
    print(pred.head(args.head))

    e = mlp.binary_cross_entropy(y_pred, Y.to_numpy(), e=1e-15)
    print("Binary cross entropy: ", e)