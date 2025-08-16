import numpy as np
from monkeyframe import DataFrame

def test_inner_merge():
    left = DataFrame({"city": np.array(["A","B"]), "val": np.array([1,2])})
    right = DataFrame({"city": np.array(["B","C"]), "val2": np.array([10,20])})
    result = left.merge(right, on="city", how="inner")
    assert list(result["city"]) == ["B"]

def test_left_merge():
    left = DataFrame({"city": np.array(["A","B","C"]), "val": np.array([1,2,3])})
    right = DataFrame({"city": np.array(["A","B"]), "val2": np.array([10,20])})
    result = left.merge(right, on="city", how="left")
    assert result["val2"][2] == -1

def test_outer_merge():
    left = DataFrame({"city": np.array(["A"]), "val": np.array([1])})
    right = DataFrame({"city": np.array(["B"]), "val2": np.array([10])})
    result = left.merge(right, on="city", how="outer")
    assert len(result["city"]) == 2
