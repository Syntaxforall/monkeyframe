import numpy as np
from .dataframe import DataFrame
from .series import Series

def train_test_split(df, target, test_size=0.2, shuffle=True, random_state=None):
    if random_state:
        np.random.seed(random_state)

    n = len(df)
    idx = np.arange(n)
    
    if shuffle:
        np.random.shuffle(idx)
    
    split = int(n * (1 - test_size))
    train_idx, test_idx = idx[:split], idx[split:]
    
    X_cols = [c for c in df.columns if c != target]
    X_df = df[X_cols]
    X_train = X_df.iloc(train_idx)
    X_test = X_df.iloc(test_idx)
    
    y_series = df[target]
    y_train = y_series.iloc(train_idx)
    y_test = y_series.iloc(test_idx)
    
    return X_train, X_test, y_train, y_test

def kfold(n_samples, n_splits=5, shuffle=True, random_state=None):
    if random_state:
        np.random.seed(random_state)

    idx = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(idx)

    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[:n_samples % n_splits] += 1

    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        test_idx = idx[start:stop]
        train_idx = np.concatenate([idx[:start], idx[stop:]])
        yield train_idx, test_idx
        current = stop

def shuffle_df(df, random_state=None):
    if random_state:
        np.random.seed(random_state)

    idx = np.arange(len(df))
    np.random.shuffle(idx)
    return df.iloc(idx)