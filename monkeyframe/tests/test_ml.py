import numpy as np
import pytest
from monkeyframe.dataframe import DataFrame
from monkeyframe.ml import train_test_split, kfold, shuffle_df

@pytest.fixture
def ml_df():
    data = {
        'A': np.arange(10),
        'B': np.arange(10) * 2,
        'y': np.array([0, 1] * 5)
    }
    return DataFrame(data)

def test_train_test_split(ml_df):
    X_train, X_test, y_train, y_test = train_test_split(ml_df, target='y', test_size=0.2, random_state=42)

    assert X_train.shape == (8, 2)
    assert X_test.shape == (2, 2)
    assert len(y_train) == 8
    assert len(y_test) == 2

def test_kfold():
    n_samples = 10
    n_splits = 5
    kf = kfold(n_samples, n_splits=n_splits)

    folds = list(kf)
    assert len(folds) == n_splits
    for train_idx, test_idx in folds:
        assert len(test_idx) == n_samples / n_splits
        assert len(train_idx) == n_samples - (n_samples / n_splits)

def test_shuffle_df(ml_df):
    shuffled_df = shuffle_df(ml_df, random_state=42)

    assert shuffled_df.shape == ml_df.shape
    # Check that the index is shuffled
    assert not np.array_equal(shuffled_df.index.data, ml_df.index.data)
    # Check that the data is shuffled along with the index
    # (This is harder to test without a .loc method, but we can check if the columns are still aligned)
    assert np.array_equal(shuffled_df['A'].data * 2, shuffled_df['B'].data)
