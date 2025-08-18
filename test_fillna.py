import numpy as np
from monkeyframe.dataframe import DataFrame

# Create a DataFrame with missing values
df = DataFrame({
    'A': np.array([1.0, 2.0, np.nan, 4.0, 5.0]),
    'B': np.array(['foo', 'bar', 'baz', 'qux', None], dtype=object),
    'C': np.array([np.nan, 1, 2, 3, 4])
})

print("Original DataFrame:")
print(df)

# Test 1: Fill NaN in a float column with a scalar value
df1 = df.fillna(0.0, cols='A')
print("\nAfter filling 'A' with 0.0:")
print(df1)

# Test 2: Fill None in an object column with a scalar value
df2 = df.fillna('missing', cols='B')
print("\nAfter filling 'B' with 'missing':")
print(df2)

# Test 3: Fill all columns with a single scalar value
df3 = df.fillna(-1)
print("\nAfter filling all columns with -1:")
print(df3)

# Test 4: Fill different columns with different values using a dictionary
fill_values = {'A': 99.9, 'B': 'filled', 'C': -100}
df4 = df.fillna(fill_values)
print("\nAfter filling with a dictionary:")
print(df4)

# Test that original dataframe is not modified
print("\nOriginal DataFrame after all operations:")
print(df)
