import numpy as np
import pytest
from monkeyframe.dataframe import DataFrame
import io
import sys

def test_info_runs():
    """Tests that the info() method runs without errors and prints something."""
    df = DataFrame({
        'A': np.array([1.0, 2.0, np.nan]),
        'B': np.array(['foo', 'bar', None], dtype=object)
    })

    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = captured_output = io.StringIO()

    df.info()

    # Restore stdout
    sys.stdout = old_stdout

    # Check that something was printed
    assert len(captured_output.getvalue()) > 0

@pytest.mark.skip(reason="Not implemented on new architecture yet")
def test_fillna():
    """Tests the enhanced fillna() method."""
    df = DataFrame({
        'A': np.array([1.0, 2.0, np.nan, 4.0, 5.0]),
        'B': np.array(['foo', 'bar', 'baz', 'qux', None], dtype=object),
        'C': np.array([np.nan, 1, 2, 3, 4])
    })

    # Test 1: Fill NaN in a float column with a scalar value
    df1 = df.fillna(0.0, cols='A')
    assert df1['A'][2] == 0.0

    # Test 2: Fill None in an object column with a scalar value
    df2 = df.fillna('missing', cols='B')
    assert df2['B'][4] == 'missing'

    # Test 3: Fill all columns with a single scalar value
    df3 = df.fillna(-1)
    assert df3['A'][2] == -1.0
    assert df3['B'][4] == -1
    assert df3['C'][0] == -1.0

    # Test 4: Fill different columns with different values using a dictionary
    fill_values = {'A': 99.9, 'B': 'filled', 'C': -100}
    df4 = df.fillna(fill_values)
    assert df4['A'][2] == 99.9
    assert df4['B'][4] == 'filled'
    assert df4['C'][0] == -100.0

    # Test that original dataframe is not modified
    assert np.isnan(df['A'][2])
    assert df['B'][4] is None
