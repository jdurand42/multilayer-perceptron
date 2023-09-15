import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score, accuracy_score
from sklearn.metrics import confusion_matrix

from utils import *
from MLP import MultiLayerPerceptron

import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--data_path_train', type=str, default='../data/df_train.csv',
                    help='path to csv file containing data for train')
parser.add_argument('--data_path_test', type=str, default='../data/df_test.csv',
                    help='path to csv file containing data for train')
parser.add_argument('-e','--early_stopping', type=int, default = None,
                    help='number of epochs needed for early stopping')
parser.add_argument('-p','--precision', type=int, default = 5,
                    help='precision for early stopping')
parser.add_argument('--export_path', type=str, default = "mlp.pkl",
                    help='Output path for model pkl')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--hidden_layers_number', type=int, default=2)
parser.add_argument('--hidden_layers_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--no_print', action="store_false")
parser.add_argument("--verbose", action="store_true")

if __name__ == "__main__":
    args = parser.parse_args()

    df_train = pd.read_csv(args.data_path_train)
    df_test = pd.read_csv(args.data_path_test)

    # df = get_data(args.data_path, headers=["id", "diagnosis"])
    X_train, Y_train = get_X_Y(df_train, labels=["diagnosis"], drops=[])
    X_test, Y_test = get_X_Y(df_test, labels=["diagnosis"], drops=[])

    Y_train = labelize_Y(Y_train, y_label="diagnosis", value="M")
    Y_test = labelize_Y(Y_test, y_label="diagnosis", value="M")
    

    # raw = X.copy()

    print(df_train.describe())

    print(X_train.head(2))
    print(Y_train.head(2))
    print("shape:", X_train.shape)

    print(X_test.head(2))

    stds = stds(X_train)
    means = means(X_train)
    X_train = zscore(X_train, stds, means)
    X_test = zscore(X_test, stds, means)

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
    e = binary_cross_entropy(y_pred, Y_test.to_numpy())
    print("Binary cross entropy: ", e)
    print("score: ", accuracy_score(Y_test.to_numpy(), y_pred))
    # e = binary_cross_entropy(y_pred, Y_test.to_numpy())
    # print(e)





