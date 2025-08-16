import numpy as np
from numba import njit

# ==========================
# Low-level kernels
# ==========================
@njit
def _groupby_agg(group_ids, numeric_data, num_groups, agg_type):
    # agg_type: 0=mean, 1=sum, 2=count
    if numeric_data.size == 0:
        counts = np.zeros(num_groups)
        for g in group_ids:
            counts[g] += 1
        return counts.reshape(-1, 1)

    n_rows, n_cols = numeric_data.shape
    sums = np.zeros((num_groups, n_cols))
    counts = np.zeros(num_groups)

    for i in range(n_rows):
        g = group_ids[i]
        counts[g] += 1.0
        if agg_type != 2:
            for j in range(n_cols):
                sums[g, j] += numeric_data[i, j]
        else:
            for j in range(n_cols):
                sums[g, j] += 1.0

    if agg_type == 0:  # mean
        for g in range(num_groups):
            if counts[g] > 0:
                for j in range(n_cols):
                    sums[g, j] /= counts[g]
        return sums
    elif agg_type == 1:  # sum
        return sums
    else:  # count
        return counts.reshape(-1, 1)

# ==========================
# DataFrame
# ==========================
class DataFrame:
    """
    Minimal, NumPy/Numba-native DataFrame aimed at ML workflows.
    - Columnar store of equal-length 1D np.ndarray
    - Zero-copy column selection when possible
    - Vectorized ops; simple groupby; join; encoders; describe
    """
    # ---- Construction ----
    def __init__(self, data):
        if isinstance(data, dict):
            cols = list(data.keys())
            arrays = {c: np.asarray(data[c]) for c in cols}
        else:
            raise TypeError("DataFrame(data) expects a dict of column->array")

        lengths = {len(v) for v in arrays.values()}
        if len(lengths) != 1:
            raise ValueError("All columns must have the same length")
        self.data = arrays
        self.columns = list(arrays.keys())
        self.length = next(iter(lengths))

    @classmethod
    def from_arrays(cls, columns, arrays):
        return cls({c: np.asarray(a) for c, a in zip(columns, arrays)})

    def copy(self):
        return DataFrame({c: v.copy() for c, v in self.data.items()})

    # ---- Introspection ----
    @property
    def shape(self):
        return (self.length, len(self.columns))

    @property
    def dtypes(self):
        return {c: v.dtype for c, v in self.data.items()}

    def to_numpy(self, cols=None):
        if cols is None: cols = self.columns
        return np.column_stack([self.data[c] for c in cols])

    # ---- Display ----
    def __repr__(self):
        max_rows = 10
        index_width = len(str(max(self.length - 1, 0)))
        col_widths = {}
        for col in self.columns:
            vals = self.data[col]
            w = len(str(col))
            if self.length:
                w = max(w, *(len(str(vals[i])) for i in range(min(self.length, max_rows))))
            col_widths[col] = min(max(w, 6), 60)

        header = " " * (index_width + 2) + "  ".join(f"{col:<{col_widths[col]}}" for col in self.columns)
        lines = [header]
        if self.length > max_rows:
            display_rows = list(range(5)) + list(range(self.length - 5, self.length))
            skip_marker = True
        else:
            display_rows = list(range(self.length))
            skip_marker = False

        for i in display_rows:
            row = f"{i:<{index_width}}  " + "  ".join(
                f"{self.data[col][i]:<{col_widths[col]}}" if self.data[col].dtype.kind in "OUS"
                else f"{self.data[col][i]:>{col_widths[col]}}"
                for col in self.columns
            )
            lines.append(row)
            if skip_marker and i == 4:
                lines.append(" " * (index_width + 2) + "...")
        return "\n".join(lines)

    # ---- Core selection/indexing ----
    def __getitem__(self, key):
        if isinstance(key, str):
            return self.data[key]
        elif isinstance(key, list):
            return self.select(key)
        elif isinstance(key, np.ndarray) and key.dtype == bool:
            return self.filter(key)
        else:
            raise TypeError("Use df['col'], df[['c1','c2']], or boolean mask")

    def select(self, cols):
        if isinstance(cols, str): cols = [cols]
        return DataFrame({c: self.data[c] for c in cols})

    def filter(self, mask):
        mask = np.asarray(mask, dtype=bool)
        if mask.shape[0] != self.length:
            raise ValueError("Boolean mask length mismatch")
        return DataFrame({c: v[mask] for c, v in self.data.items()})

    def assign(self, **newcols):
        out = {**self.data}
        for k, v in newcols.items():
            v = np.asarray(v)
            if v.shape[0] != self.length:
                raise ValueError("Assigned column length mismatch")
            out[k] = v
        return DataFrame(out)

    def drop(self, cols):
        if isinstance(cols, str): cols = [cols]
        return DataFrame({c: v for c, v in self.data.items() if c not in cols})

    def head(self, n=5):
        idx = slice(0, min(n, self.length))
        return DataFrame({c: v[idx] for c, v in self.data.items()})

    def tail(self, n=5):
        n = min(n, self.length)
        idx = slice(self.length - n, self.length)
        return DataFrame({c: v[idx] for c, v in self.data.items()})

    def sort_values(self, by, ascending=True):
        if isinstance(by, str): by = [by]
        idx = np.arange(self.length)
        # stable multi-key
        for key in reversed(by):
            order = np.argsort(self.data[key], kind="mergesort")
            idx = idx[order]
        if not ascending:
            idx = idx[::-1]
        return DataFrame({c: v[idx] for c, v in self.data.items()})

    def astype(self, **types):
        out = {**self.data}
        for c, t in types.items():
            out[c] = self.data[c].astype(t, copy=False)
        return DataFrame(out)

    def fillna(self, value, cols=None):
        if cols is None: cols = self.columns
        out = {**self.data}
        for c in cols:
            col = out[c]
            if col.dtype.kind == "f":
                m = np.isnan(col)
                if m.any():
                    out[c] = np.where(m, value, col)
        return DataFrame(out)

    # ---- Arithmetic & columnwise ops (broadcast on scalars / columns) ----
    def _binary_op(self, other, op, on=None):
        out = {}
        if on is None:
            # scalar or dict of scalars
            if np.isscalar(other):
                for c, v in self.data.items():
                    out[c] = op(v, other) if np.issubdtype(v.dtype, np.number) else v
            elif isinstance(other, dict):
                for c, v in self.data.items():
                    if c in other and np.issubdtype(v.dtype, np.number):
                        out[c] = op(v, other[c])
                    else:
                        out[c] = v
            else:
                raise TypeError("Unsupported operand")
        else:
            # op on selected numeric columns
            if np.isscalar(other):
                for c, v in self.data.items():
                    out[c] = op(v, other) if (c in on and np.issubdtype(v.dtype, np.number)) else v
            else:
                raise TypeError("Non-scalar with `on=` not supported")
        return DataFrame(out)

    def add(self, other, on=None): return self._binary_op(other, np.add, on)
    def sub(self, other, on=None): return self._binary_op(other, np.subtract, on)
    def mul(self, other, on=None): return self._binary_op(other, np.multiply, on)
    def div(self, other, on=None): return self._binary_op(other, np.divide, on)

    # ---- Stats / describe ----
    def describe(self):
        numeric_cols = [c for c in self.columns if np.issubdtype(self.data[c].dtype, np.number)]
        if not numeric_cols:
            return DataFrame({"stat": np.array([])})
        X = self.to_numpy(numeric_cols)
        res = {
            "stat": np.array(["count", "mean", "std", "min", "25%", "50%", "75%", "max"], dtype=object)
        }
        count = np.sum(~np.isnan(X), axis=0)
        mean = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0, ddof=1)
        q = np.nanpercentile(X, [0, 25, 50, 75, 100], axis=0)
        table = np.vstack([count, mean, std, q])
        for j, c in enumerate(numeric_cols):
            res[c] = table[:, j]
        return DataFrame(res)

    # ---- Value counts ----
    def value_counts(self, col):
        vals, counts = np.unique(self.data[col], return_counts=True)
        return DataFrame({"value": vals, "count": counts})

    # ---- Groupby (compact) ----
    def groupby(self, by_cols, agg="mean"):
        if isinstance(by_cols, str): by_cols = [by_cols]
        # encode keys -> codes
        code_cols, uniques_per_col = [], []
        for col in by_cols:
            u, codes = np.unique(self.data[col], return_inverse=True)
            uniques_per_col.append(u)
            code_cols.append(codes.astype(np.int64))
        codes_matrix = np.column_stack(code_cols) if len(code_cols) > 1 else code_cols[0].reshape(-1, 1)

        unique_rows, group_ids = np.unique(codes_matrix, axis=0, return_inverse=True)
        num_groups = unique_rows.shape[0]

        numeric_cols = [c for c in self.columns if c not in by_cols and np.issubdtype(self.data[c].dtype, np.number)]
        numeric_data = np.column_stack([self.data[c] for c in numeric_cols]) if numeric_cols else np.empty((self.length, 0))

        agg_map = {"mean": 0, "sum": 1, "count": 2}
        agg_type = agg_map.get(agg, 0)

        result_numeric = _groupby_agg(group_ids.astype(np.int64), numeric_data, num_groups, agg_type)

        result_data = {}
        for k, col in enumerate(by_cols):
            result_data[col] = uniques_per_col[k][ unique_rows[:, k] ]

        if numeric_cols:
            for j, c in enumerate(numeric_cols):
                result_data[c] = result_numeric[:, j]
        else:
            result_data["count"] = result_numeric[:, 0]

        return DataFrame(result_data)

    # ---- One-hot / label encode / standardize ----
    def one_hot(self, cols, drop_first=False, prefix=None):
        if isinstance(cols, str): cols = [cols]
        out = {**self.data}
        for c in cols:
            cats = np.unique(self.data[c])
            use = cats[1:] if drop_first else cats
            for k in use:
                name = f"{prefix or c}_{k}"
                out[name] = (self.data[c] == k).astype(np.int8)
        # keep original? For ML usually drop original
        for c in cols:
            out.pop(c, None)
        return DataFrame(out)

    def label_encode(self, cols):
        if isinstance(cols, str): cols = [cols]
        out = {**self.data}
        for c in cols:
            u, inv = np.unique(self.data[c], return_inverse=True)
            out[c] = inv.astype(np.int32)
        return DataFrame(out)

    def standardize(self, cols=None, with_mean=True, with_std=True, eps=1e-12):
        if cols is None:
            cols = [c for c in self.columns if np.issubdtype(self.data[c].dtype, np.number)]
        out = {**self.data}
        for c in cols:
            x = self.data[c].astype(np.float64, copy=False)
            if with_mean:
                m = np.nanmean(x)
                x = x - m
            if with_std:
                s = np.nanstd(x, ddof=0)
                x = x / (s + eps)
            out[c] = x
        return DataFrame(out)

    def select_dtypes(self, include=None, exclude=None):
        include = set(include or [])
        exclude = set(exclude or [])
        keep = []
        for c, v in self.data.items():
            k = v.dtype.kind
            if include and k not in include: 
                continue
            if exclude and k in exclude:
                continue
            if not include and not exclude:
                keep.append(c)
            else:
                keep.append(c)
        return self.select(keep)

    # ---- Concatenate rows ----
    @staticmethod
    def concat(dfs):
        if not dfs: 
            return DataFrame({})
        cols = dfs[0].columns
        for df in dfs:
            if df.columns != cols:
                raise ValueError("All DataFrames must have same columns for concat")
        stacked = {c: np.concatenate([df.data[c] for df in dfs]) for c in cols}
        return DataFrame(stacked)

    # ---- Simple merge (single key, inner/left) ----
    def merge(self, other, on, how="inner", suffixes=("_x", "_y")):
        if isinstance(on, list) and len(on) != 1:
            raise NotImplementedError("Only single-key merges supported for now")
        if isinstance(on, list):
            on = on[0]
        
        left_key = self.data[on]
        right_key = other.data[on]

        # Sort right key for searchsorted

        r_order = np.argsort(right_key, kind="mergesort")
        rk_sorted = right_key[r_order]

        # Find positions in right for each left key
        left_pos = np.searchsorted(rk_sorted, left_key, side="left")
        right_match = np.full(self.length, -1, dtype=np.int64)
        mask = np.zeros_like(left_pos, dtype=bool)
        in_bounds = left_pos < rk_sorted.size
        mask[in_bounds] = rk_sorted[left_pos[in_bounds]] == left_key[in_bounds]
        right_match[mask] = r_order[left_pos[mask]]
        # --- Build join index sets ---
        if how == "inner":
            left_idx = np.where(right_match >= 0)[0]
            right_idx = right_match[left_idx]
        elif how == "left":
            left_idx = np.arange(self.length)
            right_idx = right_match
        elif how == "right":
            #Symmetry: swap left/right and re-call
            return other.merge(self, on=on, how="left", suffixes=suffixes[::-1])
        elif how == "outer":
            # Left side
            left_only_mask = right_match < 0
            left_idx = np.arange(self.length)
            right_idx = right_match
            # Right-only rows
            right_unmatched = np.setdiff1d(np.arange(len(right_key)), right_match[right_match >= 0], assume_unique=True)
            if right_unmatched.size:
                left_idx = np.concatenate([left_idx, np.full(right_unmatched.size, -1)])
                right_idx = np.concatenate([right_idx, right_unmatched])
        else:
            raise ValueError("how must be one of {'inner','left','right','outer'}")
        # --- Build output ---
        out = {}
        # Left table columns
        for c in self.columns:
            if c == on:
                out[c] = self._merge_col_values(c, left_idx, right_idx, other, is_left=True, on=on)
            else:
                name = c + suffixes[0] if c in other.columns else c
                out[name] = self._merge_col_values(c, left_idx, right_idx, other, is_left=True, on=on)
        # Right table columns
        for c in other.columns:
            if c == on:
                continue
            name = c + suffixes[1] if c in self.columns else c
            out[name] = self._merge_col_values(c, left_idx, right_idx, other, is_left=False, on=on)
        return DataFrame(out)
    def _merge_col_values(self, col, left_idx, right_idx, other, is_left, on):
        if is_left:
            arr = self.data[col]
            out_col = np.empty(len(left_idx), dtype=arr.dtype)
            if arr.dtype.kind in "f":
                out_col[:] = np.nan
            elif np.issubdtype(arr.dtype, np.integer):
                out_col[:] = -1
            else:
                out_col[:] = ""
            mask = left_idx >= 0
            out_col[mask] = arr[left_idx[mask]]
        else:
            arr = other.data[col]
            out_col = np.empty(len(right_idx), dtype=arr.dtype)
            if arr.dtype.kind in "f":
                out_col[:] = np.nan
            elif np.issubdtype(arr.dtype, np.integer):
                out_col[:] = -1
            else:
                out_col[:] = ""
            mask = right_idx >= 0
            out_col[mask] = arr[right_idx[mask]]
        return out_col
# ==========================
# ML helpers
# ==========================
def _check_random_state(seed):
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    return np.random.RandomState(seed)

def train_test_split(df, test_size=0.2, shuffle=True, stratify=None, random_state=None):
    n = df.length
    rs = _check_random_state(random_state)
    idx = np.arange(n)

    if stratify is None:
        if shuffle:
            rs.shuffle(idx)
        split = int(n * (1 - test_size))
        train_idx, test_idx = idx[:split], idx[split:]
    else:
        y = np.asarray(df.data[stratify])
        # stratified split per class
        train_idx, test_idx = [], []
        for cls in np.unique(y):
            cls_idx = np.where(y == cls)[0]
            if shuffle:
                rs.shuffle(cls_idx)
            split = int(len(cls_idx) * (1 - test_size))
            train_idx.append(cls_idx[:split])
            test_idx.append(cls_idx[split:])
        train_idx = np.concatenate(train_idx)
        test_idx = np.concatenate(test_idx)
        if shuffle:
            rs.shuffle(train_idx); rs.shuffle(test_idx)

    def take(indexer):
        return DataFrame({c: v[indexer] for c, v in df.data.items()})

    return take(train_idx), take(test_idx)

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

# ==========================
# Quick demo
# ==========================
if __name__ == "__main__":
    df = DataFrame({
        "city": np.array(["A","A","B","B","C","C","A","B"]),
        "x1":   np.array([1.,2.,3.,4.,5.,6.,7.,8.]),
        "x2":   np.array([10,20,30,40,50,60,70,80]),
        "y":    np.array([0,1,0,1,0,1,0,1])
    })

    print("DF:\n", df)
    print("\nDescribe:\n", df.describe())

    print("\nGroupby city mean:\n", df.groupby("city", agg="mean"))

    df2 = df.one_hot("city", drop_first=True)
    print("\nOne-hot (drop_first):\n", df2)

    train, test = train_test_split(df2, test_size=0.25, stratify="y", random_state=42)
    print("\nTrain:\n", train)
    print("\nTest:\n", test)

    left = DataFrame({
    "city": np.array(["A","B","C"]),
    "x1": np.array([1,2,3])})
    right = DataFrame({
    "city": np.array(["A","B"]),
    "z": np.array([100,200])})
    print("Inner:\n", left.merge(right, on="city", how="inner"))
    print("\nLeft:\n", left.merge(right, on="city", how="left"))
    print("\nRight:\n", left.merge(right, on="city", how="right"))
    print("\nOuter:\n", left.merge(right, on="city", how="outer"))
