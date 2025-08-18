import numpy as np
from monkeyframe.dataframe import DataFrame

def test_standardize():
    df = DataFrame({
        'A': np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        'B': np.array([10.0, 20.0, 30.0, 40.0, 50.0]),
        'C': np.array(['foo', 'bar', 'baz', 'qux', 'quux'], dtype=object)
    })

    standardized_df = df.standardize(cols=['A', 'B'])

    # Check if non-numeric column is untouched
    assert np.array_equal(standardized_df['C'], df['C'])

    # Check mean and std of standardized columns
    mean_A = np.mean(standardized_df['A'])
    std_A = np.std(standardized_df['A'])

    mean_B = np.mean(standardized_df['B'])
    std_B = np.std(standardized_df['B'])

    assert np.isclose(mean_A, 0.0)
    assert np.isclose(std_A, 1.0)

    assert np.isclose(mean_B, 0.0)
    assert np.isclose(std_B, 1.0)

    # Test with default columns (all numeric)
    standardized_df_all = df.standardize()
    mean_A_all = np.mean(standardized_df_all['A'])
    std_A_all = np.std(standardized_df_all['A'])
    assert np.isclose(mean_A_all, 0.0)
    assert np.isclose(std_A_all, 1.0)
