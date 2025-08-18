import numpy as np
import pytest
import io
import sys
from monkeyframe.dataframe import DataFrame

@pytest.fixture
def numeric_df():
    data = {
        'A': [1, 2, 3, 4, 5],
        'B': [1.1, 2.2, 3.3, 4.4, 5.5],
        'C': [10, 20, 30, 40, 50]
    }
    return DataFrame(data)

def test_info(numeric_df):
    old_stdout = sys.stdout
    sys.stdout = captured_output = io.StringIO()

    numeric_df.info()

    sys.stdout = old_stdout

    output = captured_output.getvalue()
    assert "<class 'monkeyframe.dataframe.DataFrame'>" in output
    assert "Data columns (total 3 columns)" in output
    assert "A" in output
    assert "B" in output
    assert "C" in output

def test_describe(numeric_df):
    desc_df = numeric_df.describe()

    assert isinstance(desc_df, DataFrame)
    assert desc_df.shape == (8, 3)
    assert list(desc_df.index.data) == ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']

    # Check some values
    assert np.isclose(desc_df['A']['mean'], 3.0)
    assert np.isclose(desc_df['B']['std'], np.std(np.array([1.1, 2.2, 3.3, 4.4, 5.5])))
    assert np.isclose(desc_df['C']['max'], 50)
