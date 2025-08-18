import numpy as np

class Index:
    def __init__(self, data, name=None):
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        self.data = data
        self.name = name

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f"Index({self.data.tolist()}, name='{self.name}')"

    def copy(self):
        return Index(self.data.copy(), name=self.name)

    def equals(self, other):
        if not isinstance(other, Index):
            return False
        return np.array_equal(self.data, other.data)
