import numpy as np

# A DataFrame class for holding and manipulating columnar data.
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
            arrays = {c: np.asarray(data[c]) for c in data}
        else:
            raise TypeError("DataFrame(data) expects a dict of column->array")
        lengths = {len(v) for v in arrays.values()}
        if len(lengths) != 1:
            raise ValueError("All columns must have same length")
        self.data = arrays
        self.columns = list(arrays.keys())
        self.length = next(iter(lengths))

    @classmethod
    def from_arrays(cls, columns, arrays):
        return cls({c: np.asarray(a) for c, a in zip(columns, arrays)})

    def copy(self):
        """Returns a deep copy of the DataFrame."""
        return DataFrame({c: v.copy() for c, v in self.data.items()})

    # ---- Introspection ----
    @property
    def shape(self):
        """Returns the dimensions of the DataFrame as a tuple (rows, columns)."""
        return (self.length, len(self.columns))

    @property
    def dtypes(self):
        """Returns a dictionary of column names and their data types."""
        return {c: v.dtype for c, v in self.data.items()}

    def to_numpy(self, cols=None):
        if cols is None: cols = self.columns
        return np.column_stack([self.data[c] for c in cols])

    # ---- Display ----
    def __repr__(self):
        """Returns a string representation of the DataFrame."""
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
        """Returns a new DataFrame with new columns or replaced existing ones."""
        out = {**self.data}
        for k, v in newcols.items():
            v = np.asarray(v)
            if v.shape[0] != self.length:
                raise ValueError("Assigned column length mismatch")
            out[k] = v
        return DataFrame(out)

    def drop(self, cols):
        """Returns a new DataFrame with specified columns removed."""
        if isinstance(cols, str): cols = [cols]
        return DataFrame({c: v for c, v in self.data.items() if c not in cols})

    def head(self, n=5):
        """Returns the first n rows."""
        idx = slice(0, min(n, self.length))
        return DataFrame({c: v[idx] for c, v in self.data.items()})

    def tail(self, n=5):
        """Returns the last n rows."""
        n = min(n, self.length)
        idx = slice(self.length - n, self.length)
        return DataFrame({c: v[idx] for c, v in self.data.items()})

    def sort_values(self, by, ascending=True):
        if isinstance(by, str): by = [by]
        idx = np.arange(self.length)
        for col in reversed(by):
            idx = idx[np.argsort(self.data[col][idx], kind='mergesort')]
        if not ascending:
            idx = idx[::-1]
        return DataFrame({c: v[idx] for c, v in self.data.items()})

    def astype(self, **types):
        """Casts columns to a specified data type."""
        out = {**self.data}
        for c, t in types.items():
            out[c] = self.data[c].astype(t, copy=False)
        return DataFrame(out)

    def fillna(self, value, cols=None):
        """Fills NaN values in numeric columns."""
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
        """Helper for binary operations."""
        out = {}
        if on is None:
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
        """Generates descriptive statistics for numeric columns."""
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

    def iloc(self, idx):
        if isinstance(idx, int):
            idx = [idx]
        return DataFrame({c: self.data[c][idx] for c in self.columns})

    # ---- Value counts ----
    def value_counts(self, col):
        """Returns a DataFrame with counts of unique values for a column."""
        vals, counts = np.unique(self.data[col], return_counts=True)
        return DataFrame({"value": vals, "count": counts})

    # ---- Concatenate rows ----
    @staticmethod
    def concat(dfs):
        if not dfs: return DataFrame({})
        cols = dfs[0].columns
        for df in dfs:
            if df.columns != cols:
                raise ValueError("All DataFrames must have same columns for concat")
        stacked = {c: np.concatenate([df.data[c] for df in dfs]) for c in cols}
        return DataFrame(stacked)

    def select_dtypes(self, include=None, exclude=None):
        """Returns a new DataFrame with a subset of columns based on their data types."""
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