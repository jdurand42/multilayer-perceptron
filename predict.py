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

parser.add_argument('--data_path', '-d', type=str, default='./data/df_test.csv',
                    help='path to csv file containing data')
parser.add_argument('--model_path', '-m', type=str, default="mlp.pkl")
parser.add_argument('--raw', action="store_true")
parser.add_argument('--no_print', action="store_false")
parser.add_argument('--head', type=int, default=3)
parser.add_argument('--export', '-e', type=str, default=None)
parser.add_argument('-f', '--features', type=list, default=features)
parser.add_argument('--softmax', action="store_true",
                    help="Perform one hot encoding and use softmax zith an output layer size of 2")

if __name__ == "__main__":
    args = parser.parse_args()

    print(args.features)
    df = get_data_pred(args.data_path)
    X = df[features]
    Y = []
    if 'diagnosis' in df.columns:
        Y = df[['diagnosis']]
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
    pred = []
    # print(y_pred)
    if (len(y_pred.shape) == 1):
        for i in range(0, len(y_pred)):
            if y_pred[i] == 0:
                pred.append("B")
            else:
                pred.append("M")
        pred = pd.DataFrame(data={'diagnosis': pred})
            
    else:
        pred = unencode(y_pred, values=["B", "M"], label="diagnosis")
    print(pred.head(args.head))

    # if len(Y) > 0:
    e = mlp.binary_cross_entropy(labelize_Y(pred).to_numpy(), labelize_Y(Y).to_numpy())
    print("Binary cross entropy: ", e)

    if args.export is not None:
        pred.to_csv(args.export)

