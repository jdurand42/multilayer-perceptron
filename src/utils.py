import pandas as pd 
import numpy as np
import math

def stds(X):
    stds = []
    for feature in X.columns:
        stds.append(X[feature].to_numpy().std())
    return stds
        # means.append(X[feature].to_numpy().mean())

def means(X):
    means = []
    for feature in X.columns:
        means.append(X[feature].to_numpy().mean())
    return means

def zscore(X, stds, means):
    i = 0
    for feature in X.columns:
        X[feature] = (X[feature] - means[i]) / stds[i]
        i += 1
    return X

def get_data(data_path, headers=["id", "diagnosis"], features_len=30):
    for i in range(0, features_len):
        headers.append(f'F{i}')
    df = pd.read_csv(data_path, names=headers)
    return df

def get_X_Y(df, labels=["diagnosis"], drops=['id']):
    Y = df[labels]
    X = df.drop(labels=labels+drops, axis=1, inplace=False)
    return X, Y

def labelize_Y(Y, y_label="diagnosis", value="M"):
    Y = Y[y_label].apply(lambda x: 1 if x == value else 0)
    return Y

def unlabelize_Y(Y, y_label="diagnosis", values=("B", "M")):
    Y = Y[y_label].apply(lambda x: values[1] if x == 1 else values[0])
    return Y

# def process_data(data_path):
#     df = get_data(data_path)
#     X,Y = get_X_Y(df)
#     Y = labelize_Y(Y)
#     return X, Y

def normalize(X):
    means = means(X)
    stds = stds(X)
    return zscore(X, stds, means)

def binary_cross_entropy(p, y, e=1e-15):
    r = 0
    for i in range(0, len(p)):
        r += (y[i] * math.log(p[i]+e)) + ((1 - y[i]) * math.log(1 - p[i]+e))
    r = -r / len(p)
    return r