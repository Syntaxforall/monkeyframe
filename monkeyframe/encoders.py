import numpy as np
from .dataframe import DataFrame

def _one_hot_method(df, cols, drop_first=False, prefix=None):
    if isinstance(cols, str): cols = [cols]
    out = {**df.data}
    for c in cols:
        cats = np.unique(df.data[c])
        use = cats[1:] if drop_first else cats
        for k in use:
            name = f"{prefix or c}_{k}"
            out[name] = (df.data[c] == k).astype(np.int8)
    for c in cols:
        out.pop(c, None)
    return DataFrame(out)

def _label_encode_method(df, cols):
    if isinstance(cols, str): cols = [cols]
    out = {**df.data}
    for c in cols:
        u, inv = np.unique(df.data[c], return_inverse=True)
        out[c] = inv.astype(np.int32)
    return DataFrame(out)

def _standardize_method(df, cols=None, with_mean=True, with_std=True, eps=1e-12):
    if cols is None:
        cols = [c for c in df.columns if np.issubdtype(df.data[c].dtype, np.number)]
    out = {**df.data}
    for c in cols:
        x = df.data[c].astype(np.float64, copy=False)
        if with_mean:
            m = np.nanmean(x)
            x = x - m
        if with_std:
            s = np.nanstd(x, ddof=0)
            x = x / (s + eps)
        out[c] = x
    return DataFrame(out)

DataFrame.one_hot = _one_hot_method
DataFrame.label_encode = _label_encode_method
DataFrame.standardize = _standardize_method