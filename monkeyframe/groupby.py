import numpy as np
from .dataframe import DataFrame
from .index import Index

try:
    from .monkeyframe_rust_grouper_v3 import hash_grouper_rust_v3
    RUST_GROUPER_AVAILABLE = True
except ImportError:
    RUST_GROUPER_AVAILABLE = False

def groupby(self, by):
    if not isinstance(by, str):
        raise NotImplementedError("Only single column groupby is supported for now.")

    key_col = self[by]

    if RUST_GROUPER_AVAILABLE and key_col.data.dtype.kind in 'OSU':
        import pyarrow as pa
        # The rust function expects a pyarrow array
        arrow_array = pa.Array.from_pandas(key_col.data)
        group_ids, unique_keys = hash_grouper_rust_v3(arrow_array)
        unique_keys = np.array(unique_keys)
    else:
        unique_keys, group_ids = np.unique(key_col.data, return_counts=False, return_inverse=True)

    return GroupBy(self, by, group_ids, unique_keys)

class GroupBy:
    def __init__(self, df, by, group_ids, unique_keys):
        self.df = df
        self.by = by
        self.group_ids = np.array(group_ids)
        self.unique_keys = unique_keys
        self.num_groups = len(unique_keys)

    def agg(self, agg_func):
        if agg_func not in ['mean', 'sum', 'count']:
            raise ValueError(f"Unsupported aggregation function: {agg_func}")

        numeric_cols = [c for c, s in self.df._data.items() if np.issubdtype(s.data.dtype, np.number) and c != self.by]

        if not numeric_cols:
            if agg_func == 'count':
                counts = np.bincount(self.group_ids)
                return DataFrame({'count': counts}, index=Index(self.unique_keys, name=self.by))
            else:
                return DataFrame(index=Index(self.unique_keys, name=self.by))

        data_to_agg = self.df.to_numpy(columns=numeric_cols)

        result_data = {}
        for i, col_name in enumerate(numeric_cols):
            result_col = np.empty(self.num_groups, dtype=np.float64)
            for g in range(self.num_groups):
                mask = self.group_ids == g
                if agg_func == 'mean':
                    result_col[g] = np.mean(data_to_agg[mask, i])
                elif agg_func == 'sum':
                    result_col[g] = np.sum(data_to_agg[mask, i])
                elif agg_func == 'count':
                    result_col[g] = np.sum(mask)
            result_data[col_name] = result_col

        return DataFrame(result_data, index=Index(self.unique_keys, name=self.by))

    def mean(self):
        return self.agg('mean')

    def sum(self):
        return self.agg('sum')

    def count(self):
        return self.agg('count')

# Monkey-patch
DataFrame.groupby = groupby
