import numpy as np
from .dataframe import DataFrame

def _merge_col_values(df, other, col, left_idx, right_idx, is_left):
    source_df = df if is_left else other
    arr = source_df.data[col]
    out_col = np.empty(len(left_idx), dtype=arr.dtype)

    if arr.dtype.kind in "f":
        out_col[:] = np.nan
    elif np.issubdtype(arr.dtype, np.integer):
        out_col[:] = -1
    else:
        out_col[:] = ""

    mask = left_idx >= 0 if is_left else right_idx >= 0
    indices = left_idx if is_left else right_idx

    out_col[mask] = arr[indices[mask]]
    return out_col

def _merge_method(df, other, on, how="inner", suffixes=("_x", "_y")):
    if isinstance(on, list) and len(on) != 1:
        raise NotImplementedError("Only single-key merges supported for now")
    if isinstance(on, list):
        on = on[0]

    left_key = df.data[on]
    right_key = other.data[on]

    r_order = np.argsort(right_key, kind="mergesort")
    rk_sorted = right_key[r_order]

    left_pos = np.searchsorted(rk_sorted, left_key, side="left")
    right_match = np.full(df.length, -1, dtype=np.int64)
    mask = np.zeros_like(left_pos, dtype=bool)
    in_bounds = left_pos < rk_sorted.size
    mask[in_bounds] = rk_sorted[left_pos[in_bounds]] == left_key[in_bounds]
    right_match[mask] = r_order[left_pos[mask]]

    if how == "inner":
        left_idx = np.where(right_match >= 0)[0]
        right_idx = right_match[left_idx]
    elif how == "left":
        left_idx = np.arange(df.length)
        right_idx = right_match
    elif how == "right":
        return other.merge(df, on=on, how="left", suffixes=suffixes[::-1])
    elif how == "outer":
        left_only_mask = right_match < 0
        left_idx = np.arange(df.length)
        right_idx = right_match
        right_unmatched = np.setdiff1d(np.arange(len(right_key)), right_match[right_match >= 0], assume_unique=True)
        if right_unmatched.size:
            left_idx = np.concatenate([left_idx, np.full(right_unmatched.size, -1)])
            right_idx = np.concatenate([right_idx, right_unmatched])
    else:
        raise ValueError("how must be one of {'inner','left','right','outer'}")

    out = {}
    for c in df.columns:
        if c == on:
            out[c] = _merge_col_values(df, other, c, left_idx, right_idx, is_left=True)
        else:
            name = c + suffixes[0] if c in other.columns else c
            out[name] = _merge_col_values(df, other, c, left_idx, right_idx, is_left=True)
    for c in other.columns:
        if c == on:
            continue
        name = c + suffixes[1] if c in df.columns else c
        out[name] = _merge_col_values(df, other, c, left_idx, right_idx, is_left=False)
    return DataFrame(out)

DataFrame.merge = _merge_method