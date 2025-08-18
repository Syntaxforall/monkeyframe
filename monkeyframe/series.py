import numpy as np
from .index import Index

class Series:
    def __init__(self, data, index=None, name=None):
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)

        if index is None:
            index = Index(np.arange(len(data)))
        elif not isinstance(index, Index):
            index = Index(index)

        if len(data) != len(index):
            raise ValueError("Data and index must be of the same length.")

        self.data = data
        self.index = index
        self.name = name

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, key):
        # Find the integer position of the key in the index
        try:
            pos = np.where(self.index.data == key)[0][0]
            return self.data[pos]
        except IndexError:
            raise KeyError(f"Label not found: {key}")

    def __repr__(self):
        # A simple repr for now
        return f"Series(name='{self.name}', data={self.data.tolist()}, index={self.index})"

    def notna(self):
        if np.issubdtype(self.data.dtype, np.number):
            return ~np.isnan(self.data)
        return np.array([x is not None for x in self.data])

    def copy(self):
        return Series(self.data.copy(), index=self.index.copy(), name=self.name)

    def astype(self, dtype):
        new_data = self.data.astype(dtype)
        return Series(new_data, index=self.index.copy(), name=self.name)

    def _binary_op(self, other, op):
        if isinstance(other, Series):
            # For now, assume indexes are aligned
            if not self.index.equals(other.index):
                 raise ValueError("Indexes must be equal for binary operations")
            new_data = op(self.data, other.data)
        elif np.isscalar(other):
            new_data = op(self.data, other)
        else:
            raise TypeError(f"Unsupported type for binary operation: {type(other)}")
        return Series(new_data, index=self.index.copy())

    def __add__(self, other):
        return self._binary_op(other, np.add)

    def __sub__(self, other):
        return self._binary_op(other, np.subtract)

    def __mul__(self, other):
        return self._binary_op(other, np.multiply)

    def __truediv__(self, other):
        return self._binary_op(other, np.divide)

    def __gt__(self, other):
        return self._binary_op(other, np.greater)

    def __lt__(self, other):
        return self._binary_op(other, np.less)

    def __ge__(self, other):
        return self._binary_op(other, np.greater_equal)

    def __le__(self, other):
        return self._binary_op(other, np.less_equal)

    def __eq__(self, other):
        return self._binary_op(other, np.equal)

    def __ne__(self, other):
        return self._binary_op(other, np.not_equal)

    def __and__(self, other):
        return self._binary_op(other, np.logical_and)

    def __or__(self, other):
        return self._binary_op(other, np.logical_or)

    def value_counts(self):
        unique, counts = np.unique(self.data, return_counts=True)
        # Sort by count descending
        sorted_indices = np.argsort(-counts)
        unique = unique[sorted_indices]
        counts = counts[sorted_indices]

        from .index import Index
        return Series(counts, index=Index(unique), name=self.name)

    def iloc(self, slicer):
        new_data = self.data[slicer]
        new_index = self.index.data[slicer]
        from .index import Index
        return Series(new_data, index=Index(new_index), name=self.name)
