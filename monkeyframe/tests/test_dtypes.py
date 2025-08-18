import numpy as np
import pandas as pd
from monkeyframe.dataframe import DataFrame

def test_datetime_dtype():
    dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']).to_numpy()
    df = DataFrame({'dates': dates})

    assert df['dates'].data.dtype == 'datetime64[ns]'
    assert np.array_equal(df['dates'].data, dates)

def test_nullable_integer_dtype():
    data = [1, 2, pd.NA, 4]
    df = DataFrame({'A': data})

    assert df['A'].data.dtype == 'object'
    assert df['A'].data[2] is pd.NA
