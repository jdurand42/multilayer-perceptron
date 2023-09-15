import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import *
# from MLP import MultiLayerPerceptron

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-d', '--data_path', type=str, default='../data/data.csv',
                    help='path to csv file containing data')
parser.add_argument('-r', '--ratio', type=float, default = 0.2,
                    help='ratio for test split')
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--export_path_train', type=str, default = "../data/df_train.csv",
                    help='Output path for model pkl')
parser.add_argument('-e', '--export_path_test', type=str, default = "../data/df_test.csv",
                    help='Output path for model pkl')

if __name__ == "__main__":
    args = parser.parse_args()
    df = get_data(args.data_path, headers=["id", "diagnosis"])
    X, Y = get_X_Y(df, labels=["diagnosis"], drops=['id'])
    # Y = labelize_Y(Y, y_label="diagnosis", value="M")
    raw = X.copy()

    print(df.describe())
    
    print(X.head(2))
    print(Y.head(2))

    if args.ratio != 0:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=args.ratio, random_state=args.seed)
    else:
        X_train = X.copy()
        X_test = X.copy()
        Y_train = Y.copy()
        Y_test = Y.copy()
    df_train = pd.concat([X_train, Y_train], axis=1)
    df_test = pd.concat([X_test, Y_test], axis=1)

    df_train.to_csv(f"{args.export_path_train}", index=False)
    df_test.to_csv(f"{args.export_path_test}", index=False)

    # X_train.to_csv(f"{args.export_path}/X_train.csv")
    # X_test.to_csv(f"{args.export_path}/X_test.csv") 
    # Y_test.to_csv(f"{args.export_path}/Y_test.csv")
    # Y_train.to_csv(f"{args.export_path}/Y_train.csv") 

    print(f"successfully exported the sets in '{args.export_path_train}' and '{args.export_path_test}' folder")
