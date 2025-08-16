import numpy as np
from .dataframe import DataFrame
from .utils import _check_random_state

def train_test_split(df, target, test_size=0.2, shuffle=True, random_state=None):
    n = df.length
    rs = _check_random_state(random_state)
    idx = np.arange(n)
    
    if shuffle:
        rs.shuffle(idx)
    
    split = int(n * (1 - test_size))
    train_idx, test_idx = idx[:split], idx[split:]
    
    X_cols = [c for c in df.columns if c != target]
    X_train = df.select(X_cols).iloc(train_idx)
    X_test = df.select(X_cols).iloc(test_idx)
    
    y_train = df.data[target][train_idx]
    y_test = df.data[target][test_idx]
    
    return X_train, X_test, y_train, y_test

def kfold(n_samples, n_splits=5, shuffle=True, random_state=None):
    rs = _check_random_state(random_state)
    idx = np.arange(n_samples)
    if shuffle:
        rs.shuffle(idx)
    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[: n_samples % n_splits] += 1
    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        test_idx = idx[start:stop]
        train_idx = np.concatenate([idx[:start], idx[stop:]])
        current = stop
        yield train_idx, test_idx

def shuffle_df(df, random_state=None):
    rs = _check_random_state(random_state)
    idx = np.arange(df.length)
    rs.shuffle(idx)
    return DataFrame({c: v[idx] for c, v in df.data.items()})