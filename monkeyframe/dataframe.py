import numpy as np
import pyarrow as pa

from .series import Series
from .index import Index

class DataFrame:
    def __init__(self, data=None, index=None, columns=None):
        if data is None:
            data = {}

        if isinstance(data, dict):
            if not data:
                # Empty DataFrame
                self.index = Index(index if index is not None else [])
                self.columns = columns if columns is not None else []
                self._data = {c: Series([], index=self.index, name=c) for c in self.columns}
                return

            # Infer index if not provided
            if index is None:
                first_col_data = next(iter(data.values()))
                index = Index(range(len(first_col_data)))
            elif not isinstance(index, Index):
                index = Index(index)

            self.index = index
            self._data = {}

            if columns is None:
                self.columns = list(data.keys())
            else:
                self.columns = columns

            for col in self.columns:
                col_data = data.get(col, np.nan) # Fill with NaN if col is missing in data
                self._data[col] = Series(col_data, index=self.index, name=col)

        else:
            raise TypeError("DataFrame constructor called with unsupported type")

    @property
    def shape(self):
        return (len(self.index), len(self.columns))

    def __len__(self):
        return len(self.index)

    def __repr__(self):
        # A simple repr for now
        num_rows, num_cols = self.shape
        return f"DataFrame: {num_rows} rows, {num_cols} columns"

    def __getitem__(self, key):
        if isinstance(key, str):
            if key not in self.columns:
                raise KeyError(f"Column not found: {key}")
            return self._data[key]
        elif isinstance(key, list):
            new_data = {col: self._data[col].data for col in key}
            return DataFrame(new_data, index=self.index.copy(), columns=key)
        else:
            raise TypeError(f"Unsupported key type for __getitem__: {type(key)}")

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise TypeError("Column name must be a string")

        if isinstance(value, Series):
            # Align series to dataframe index
            # This is a complex operation, for now we assume index is compatible
            if len(value) != len(self.index):
                raise ValueError("Length of series must match length of DataFrame index")
            self._data[key] = value
        elif np.isscalar(value):
            # Broadcast scalar
            self._data[key] = Series([value] * len(self), index=self.index, name=key)
        else:
            # array-like
            if len(value) != len(self.index):
                raise ValueError("Length of array-like must match length of DataFrame index")
            self._data[key] = Series(value, index=self.index, name=key)

        if key not in self.columns:
            self.columns.append(key)

    def drop(self, labels, axis=0):
        if axis == 1: # Drop columns
            new_columns = [c for c in self.columns if c not in labels]
            new_data = {c: self._data[c].data for c in new_columns}
            return DataFrame(new_data, index=self.index.copy(), columns=new_columns)
        else:
            raise NotImplementedError("Dropping rows (axis=0) is not yet implemented.")

    def assign(self, **kwargs):
        new_df = self.copy()
        for key, value in kwargs.items():
            new_df[key] = value
        return new_df

    def astype(self, dtype):
        new_data = {}
        for col_name, series in self._data.items():
            if col_name in dtype:
                new_data[col_name] = series.astype(dtype[col_name])
            else:
                new_data[col_name] = series.copy()

        return DataFrame({k: v.data for k, v in new_data.items()}, index=self.index.copy())

    def to_numpy(self, columns=None):
        if columns is None:
            columns = self.columns
        return np.column_stack([self._data[c].data for c in columns])

    def info(self):
        """Prints a concise summary of a DataFrame."""
        print(f"<class '{self.__class__.__module__}.{self.__class__.__name__}'>")
        print(f"{self.index.__class__.__name__}: {len(self.index)} entries, {self.index.data[0]} to {self.index.data[-1]}")
        print(f"Data columns (total {len(self.columns)} columns):")

        # Column details
        print(f" {'#':<3} {'Column':<15} {'Non-Null Count':<18} {'Dtype'}")
        print(f"--- {'-'*15} {'-'*18} {'-'*5}")

        for i, col in enumerate(self.columns):
            series = self._data[col]
            non_null_count = np.sum(series.notna())

            count_str = f"{non_null_count} non-null"
            print(f" {i:<3} {col:<15} {count_str:<18} {series.data.dtype}")

        # Dtypes summary
        dtype_counts = {}
        for dtype in self.dtypes.values():
            dtype_counts[str(dtype)] = dtype_counts.get(str(dtype), 0) + 1

        dtype_summary = ", ".join([f"{k}({v})" for k, v in dtype_counts.items()])
        print(f"dtypes: {dtype_summary}")

        # Memory usage
        total_mem = sum(series.data.nbytes for series in self._data.values())
        if total_mem >= 1024 * 1024:
            mem_str = f"{total_mem / (1024 * 1024):.2f}+ MB"
        elif total_mem >= 1024:
            mem_str = f"{total_mem / 1024:.2f}+ KB"
        else:
            mem_str = f"{total_mem}+ B"
        print(f"memory usage: {mem_str}")

    def describe(self):
        numeric_cols = [c for c in self.columns if np.issubdtype(self.dtypes[c], np.number)]
        if not numeric_cols:
            return DataFrame()

        stats_data = {}
        for col in numeric_cols:
            series = self._data[col]
            data = series.data[series.notna()]

            stats_data[col] = [
                len(data),
                np.mean(data),
                np.std(data),
                np.min(data),
                np.percentile(data, 25),
                np.median(data),
                np.percentile(data, 75),
                np.max(data)
            ]

        stats_index = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
        return DataFrame(stats_data, index=stats_index)

    def _binary_op(self, other, op_name):
        new_data = {}
        for col_name, series in self._data.items():
            if np.issubdtype(series.data.dtype, np.number):
                if hasattr(series, op_name):
                    op = getattr(series, op_name)
                    new_data[col_name] = op(other)
                else:
                    new_data[col_name] = series.copy()
            else:
                new_data[col_name] = series.copy()
        return DataFrame({k: v.data for k, v in new_data.items()}, index=self.index.copy())

    def add(self, other):
        return self._binary_op(other, '__add__')

    def sub(self, other):
        return self._binary_op(other, '__sub__')

    def mul(self, other):
        return self._binary_op(other, '__mul__')

    def div(self, other):
        return self._binary_op(other, '__truediv__')

    def iloc(self, slicer):
        new_index = self.index.data[slicer]
        new_data = {}
        for col_name, series in self._data.items():
            new_data[col_name] = series.data[slicer]

        return DataFrame(new_data, index=new_index)

    def filter(self, mask):
        if isinstance(mask, Series):
            mask = mask.data
        if not isinstance(mask, np.ndarray) or mask.dtype != bool:
            raise TypeError("Filter mask must be a boolean numpy array.")
        if len(mask) != len(self):
            raise ValueError("Filter mask must be of the same length as the DataFrame.")
        return self.iloc(mask)

    @property
    def dtypes(self):
        return {name: series.data.dtype for name, series in self._data.items()}

    @property
    def memory_usage(self):
        return sum(series.data.nbytes for series in self._data.values())

    def copy(self):
        return DataFrame({name: series.data.copy() for name, series in self._data.items()},
                         index=self.index.copy())

    def head(self, n=5):
        return self.iloc[:n]

    def tail(self, n=5):
        return self.iloc[-n:]

    def sort_values(self, by, ascending=True):
        if not isinstance(by, str):
            raise NotImplementedError("Only single column sort is supported for now.")

        series = self[by]
        sorted_indices = np.argsort(series.data)

        if not ascending:
            sorted_indices = sorted_indices[::-1]

        return self.iloc(sorted_indices)