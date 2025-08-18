import numpy as np
import pytest
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

@pytest.mark.skip(reason="Multi-key merge not implemented yet")
def test_multi_key_merge():
    left = DataFrame({
        'key1': np.array(['A', 'A', 'B', 'B']),
        'key2': np.array(['X', 'Y', 'X', 'Y']),
        'val1': np.array([1, 2, 3, 4])
    })
    right = DataFrame({
        'key1': np.array(['A', 'B', 'B', 'C']),
        'key2': np.array(['Y', 'X', 'Z', 'Y']),
        'val2': np.array([10, 20, 30, 40])
    })

    result = left.merge(right, on=['key1', 'key2'], how='inner')

    # Sort for consistent checking
    result = result.sort_values(by=['key1', 'key2'])

    assert len(result['key1']) == 2
    assert np.array_equal(result['key1'], np.array(['A', 'B']))
    assert np.array_equal(result['key2'], np.array(['Y', 'X']))
    assert np.array_equal(result['val1'], np.array([2, 3]))
    assert np.array_equal(result['val2'], np.array([10, 20]))
