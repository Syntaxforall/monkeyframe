import numpy as np
from .dataframe import DataFrame

def label_encode(df, col):
    series = df[col]
    unique_values, codes = np.unique(series.data, return_inverse=True)

    new_df = df.copy()
    new_df[col].data = codes
    return new_df

def one_hot(df, col):
    series = df[col]
    unique_values = np.unique(series.data)

    new_df = df.copy()
    for val in unique_values:
        new_col_name = f"{col}_{val}"
        new_df[new_col_name] = (series.data == val).astype(int)

    return new_df.drop([col], axis=1)

def standardize(df, cols=None):
    if cols is None:
        cols = [c for c in df.columns if np.issubdtype(df[c].data.dtype, np.number)]

    new_df = df.copy()
    for col in cols:
        series = new_df[col]
        data = series.data
        mean = np.mean(data)
        std = np.std(data)
        new_df[col].data = (data - mean) / std

    return new_df

# Monkey-patch
DataFrame.label_encode = label_encode
DataFrame.one_hot = one_hot
DataFrame.standardize = standardize