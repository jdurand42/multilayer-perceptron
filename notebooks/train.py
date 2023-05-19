import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score, accuracy_score
from sklearn.metrics import confusion_matrix

from utils import *
from MLP import MultiLayerPerceptron

import argparse


parser = argparse.ArgumentParser()

parser.add_argument('-d', '--data_path', type=str, default='../data/data.csv',
                    help='path to csv file containing data')
parser.add_argument('-e','--early_stopping', type=int, default = None,
                    help='number of epochs needed for early stopping')
parser.add_argument('-p','--precision', type=int, default = 5,
                    help='precision for early stopping')
parser.add_argument('--export_path', type=str, default = "mlp.pkl",
                    help='Output path for model pkl')
parser.add_argument('-s', '--split', type=float, default = 1-(350/569),
                    help='ratio for split')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--hidden_layers_number', type=int, default=2)
parser.add_argument('--hidden_layers_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--no_print', action="store_false")
parser.add_argument("--verbose", action="store_true")

if __name__ == "__main__":
    args = parser.parse_args()

    df = get_data(args.data_path, headers=["id", "diagnosis"])
    X, Y = get_X_Y(df, labels=["diagnosis"], drops=['id'])
    Y = labelize_Y(Y, y_label="diagnosis", value="M")

    raw = X.copy()

    print(df.describe())

    print(X.head(2))
    print(Y.head(2))

    stds = stds(X)
    means = means(X)

    X = zscore(X, stds, means)

    print(X.head(5))

    if args.split != 0:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=args.split)
    else:
        X_train = X.copy()
        X_test = X.copy()
        Y_train = Y.copy()
        Y_test = Y.copy()

    print("Train set size: ", len(X_train), ", test set size: ", len(X_test))
    
    mlp = MultiLayerPerceptron(seed=args.seed)
    
    for i in range(0, args.hidden_layers_number):
        mlp.add_layer(size=args.hidden_layers_size)
    mlp.add_layer(label="output_layer")

    mlp.fit(X_train.to_numpy(), Y_train.to_numpy(), verbose=args.verbose,
            epochs=args.epochs, normalization={'stds': stds, 'means': means},
            _print=args.no_print,
            X_test=X_test.to_numpy(), Y_test=Y_test.to_numpy(),
            early_stopping=args.early_stopping, precision=args.precision)

    mlp.export(path=args.export_path)

    y_pred = mlp.predict(X_test.to_numpy())
    print('score: ', accuracy_score(Y_test, y_pred))

    cm = confusion_matrix(y_pred, Y_test.to_numpy())
    print(cm)

    y_pred_raw = mlp.predict(X_test.to_numpy(), raw=True)
    e = binary_cross_entropy(y_pred_raw, Y_test.to_numpy())
    print("Binary cross entropy: ", e)
    # e = binary_cross_entropy(y_pred, Y_test.to_numpy())
    # print(e)





