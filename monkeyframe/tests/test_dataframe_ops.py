import numpy as np
import pytest
from monkeyframe.dataframe import DataFrame
from monkeyframe.series import Series

@pytest.fixture
def sample_df():
    data = {
        'A': [1, 2, 3, 4],
        'B': [1.1, 2.2, 3.3, 4.4],
        'C': ['foo', 'bar', 'baz', 'qux']
    }
    return DataFrame(data)

def test_getitem_list(sample_df):
    df = sample_df[['A', 'C']]
    assert isinstance(df, DataFrame)
    assert df.columns == ['A', 'C']
    assert df.shape == (4, 2)

def test_drop_columns(sample_df):
    df = sample_df.drop(['A', 'C'], axis=1)
    assert 'A' not in df.columns
    assert 'C' not in df.columns
    assert 'B' in df.columns
    assert df.shape == (4, 1)

def test_assign(sample_df):
    df = sample_df.assign(D=5)
    assert 'D' in df.columns
    assert np.array_equal(df['D'].data, np.array([5, 5, 5, 5]))

    df2 = sample_df.assign(E=np.arange(4))
    assert 'E' in df2.columns
    assert np.array_equal(df2['E'].data, np.arange(4))

    df3 = sample_df.assign(F=Series([10, 20, 30, 40]))
    assert 'F' in df3.columns
    assert np.array_equal(df3['F'].data, np.array([10, 20, 30, 40]))

def test_astype(sample_df):
    df = sample_df.astype({'A': float, 'B': int})
    assert df['A'].data.dtype == float
    assert df['B'].data.dtype == int

def test_binary_ops_scalar(sample_df):
    df_add = sample_df.add(10)
    assert np.array_equal(df_add['A'].data, np.array([11, 12, 13, 14]))
    assert np.array_equal(df_add['B'].data, np.array([11.1, 12.2, 13.3, 14.4]))

    df_sub = sample_df.sub(1)
    assert np.array_equal(df_sub['A'].data, np.array([0, 1, 2, 3]))

    df_mul = sample_df.mul(2)
    assert np.array_equal(df_mul['A'].data, np.array([2, 4, 6, 8]))

    df_div = sample_df.div(2)
    assert np.array_equal(df_div['A'].data, np.array([0.5, 1.0, 1.5, 2.0]))
