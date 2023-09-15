import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score, accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn.metrics import log_loss

from utils import *
from MLP import MultiLayerPerceptron
import pickle

import argparse


parser = argparse.ArgumentParser()

features=['F'+str(i) for i in range(0, 30)]

parser.add_argument('--data_path', '-d', type=str, default='../data/df_test.csv',
                    help='path to csv file containing data')
parser.add_argument('--model_path', '-m', type=str, default="mlp.pkl")
parser.add_argument('--raw', action="store_true")
parser.add_argument('--no_print', action="store_false")
parser.add_argument('--head', type=int, default=3)
parser.add_argument('--export', '-e', type=str, default=None)
parser.add_argument('-f', '--features', type=list, default=features)

if __name__ == "__main__":
    args = parser.parse_args()

    print(args.features)
    df = get_data_pred(args.data_path)
    X = df[features]
    print(X.head(5))
    # pred = Y.copy()
    # Y = labelize_Y(Y, y_label="diagnosis", value="M")

    # raw = X.copy()
    print("shape", X.shape)
    print(df.describe())

    print(X.head(2))
          
    mlp = pickle.load(open(args.model_path, "rb"))

    stds = mlp.normalization['stds']
    means = mlp.normalization['means']

    X = zscore(X, stds, means)

    # print(X.head(5))

    y_pred = mlp.predict(X.to_numpy(), raw=args.raw)
    pred = pd.DataFrame(data={'diagnosis': y_pred})
    print(pred.head(args.head))
    if args.raw is False:
        pred = unlabelize_Y(pred, y_label="diagnosis", values=("B", "M"))
        print(pred.head(args.head))
    # if args.raw is False:
    #     print("score: ", accuracy_score(Y.to_numpy(), y_pred))
    # e = mlp.binary_cross_entropy(y_pred, Y.to_numpy(), e=1e-15)
    # print("Binary cross entropy: ", e)

    if args.export is not None:
        pred.to_csv(args.export)

