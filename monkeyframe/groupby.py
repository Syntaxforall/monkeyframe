import numpy as np
from numba import njit
from .dataframe import DataFrame

# ==========================
# Numba-optimized aggregation (CPU-safe)
# ==========================

@njit
def _groupby_mean(group_ids, numeric_data, num_groups):
    n_rows, n_cols = numeric_data.shape
    sums = np.zeros((num_groups, n_cols), dtype=np.float64)
    counts = np.zeros(num_groups, dtype=np.float64)

    for i in range(n_rows):
        g = group_ids[i]
        counts[g] += 1
        for j in range(n_cols):
            sums[g, j] += numeric_data[i, j]

    for g in range(num_groups):
        if counts[g] > 0:
            for j in range(n_cols):
                sums[g, j] /= counts[g]

    return sums


@njit
def _groupby_sum(group_ids, numeric_data, num_groups):
    n_rows, n_cols = numeric_data.shape
    sums = np.zeros((num_groups, n_cols), dtype=np.float64)

    for i in range(n_rows):
        g = group_ids[i]
        for j in range(n_cols):
            sums[g, j] += numeric_data[i, j]

    return sums


@njit
def _groupby_count(group_ids, numeric_data, num_groups):
    n_rows, n_cols = numeric_data.shape

    if n_cols == 0:
        counts_1d = np.zeros(num_groups, dtype=np.float64)
        for i in range(n_rows):
            g = group_ids[i]
            counts_1d[g] += 1
        return counts_1d.reshape(num_groups, 1)

    counts = np.zeros((num_groups, n_cols), dtype=np.float64)
    for i in range(n_rows):
        g = group_ids[i]
        for j in range(n_cols):
            counts[g, j] += 1
    return counts


# ==========================
# Groupby method
# ==========================

def _groupby_method(df, by_cols, agg="mean"):
    if isinstance(by_cols, str):
        by_cols = [by_cols]

    # Encode keys -> integer codes
    code_cols, uniques_per_col = [], []
    for col in by_cols:
        u, codes = np.unique(df.data[col], return_inverse=True)
        uniques_per_col.append(u)
        code_cols.append(codes.astype(np.int64))

    codes_matrix = (
        np.column_stack(code_cols) if len(code_cols) > 1
        else code_cols[0].reshape(-1, 1)
    )

    unique_rows, group_ids = np.unique(codes_matrix, axis=0, return_inverse=True)
    num_groups = unique_rows.shape[0]

    numeric_cols = [
        c for c in df.columns
        if c not in by_cols and np.issubdtype(df.data[c].dtype, np.number)
    ]
    if numeric_cols:
        numeric_data = np.column_stack([df.data[c] for c in numeric_cols]).astype(np.float64, copy=False)
    else:
        numeric_data = np.empty((df.length, 0), dtype=np.float64)

    # Dispatch to correct aggregation kernel
    if agg == "mean":
        result_numeric = _groupby_mean(group_ids.astype(np.int64), numeric_data, num_groups)
    elif agg == "sum":
        result_numeric = _groupby_sum(group_ids.astype(np.int64), numeric_data, num_groups)
    elif agg == "count":
        result_numeric = _groupby_count(group_ids.astype(np.int64), numeric_data, num_groups)
    else:
        raise ValueError(f"Unknown aggregation: {agg}")

    # Construct result DataFrame
    result_data = {}
    for k, col in enumerate(by_cols):
        result_data[col] = uniques_per_col[k][unique_rows[:, k]]

    # Add numeric results if any
    if numeric_cols and result_numeric.shape[1] == len(numeric_cols):
        for j, c in enumerate(numeric_cols):
            result_data[c] = result_numeric[:, j]
    elif agg == "count" and not numeric_cols:
        result_data["count"] = result_numeric[:, 0]

    return DataFrame(result_data)


# ==========================
# Monkey-patch
# ==========================
DataFrame.groupby = _groupby_method
