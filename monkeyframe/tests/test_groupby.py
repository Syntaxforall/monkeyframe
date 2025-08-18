import numpy as np
import pytest
from monkeyframe.dataframe import DataFrame

@pytest.fixture
def groupby_df():
    data = {
        'key': ['A', 'B', 'A', 'B', 'A'],
        'data1': [1, 2, 3, 4, 5],
        'data2': [10.0, 20.0, 30.0, 40.0, 50.0]
    }
    return DataFrame(data)

def test_groupby_mean(groupby_df):
    grouped = groupby_df.groupby('key').mean()
    assert grouped['data1']['A'] == 3.0
    assert grouped['data1']['B'] == 3.0
    assert grouped['data2']['A'] == 30.0
    assert grouped['data2']['B'] == 30.0

def test_groupby_sum(groupby_df):
    grouped = groupby_df.groupby('key').sum()
    assert grouped['data1']['A'] == 9
    assert grouped['data1']['B'] == 6
    assert grouped['data2']['A'] == 90.0
    assert grouped['data2']['B'] == 60.0

def test_groupby_count(groupby_df):
    grouped = groupby_df.groupby('key').count()
    assert grouped['data1']['A'] == 3
    assert grouped['data1']['B'] == 2
    assert grouped['data2']['A'] == 3
    assert grouped['data2']['B'] == 2
