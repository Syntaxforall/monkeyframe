# A demonstration file for the monkeyframe library.
import numpy as np

# This import will work when the file is run as a module (e.g., `python -m monkeyframe.demo`)
# or when the monkeyframe package is installed.
from monkeyframe.dataframe import DataFrame
from monkeyframe.ml import train_test_split
from monkeyframe.encoders import _one_hot_method

if __name__ == "__main__":
    # In a real-world scenario, you would import the DataFrame and other functions directly.
    # We will use the functions as if they were part of the DataFrame object.
    DataFrame.one_hot = _one_hot_method

    df = DataFrame({
        "city": np.array(["A", "A", "B", "B", "C", "C", "A", "B"]),
        "x1": np.array([1., 2., 3., 4., 5., 6., 7., 8.]),
        "x2": np.array([10, 20, 30, 40, 50, 60, 70, 80]),
        "y": np.array([0, 1, 0, 1, 0, 1, 0, 1])
    })

    print("DF:\n", df)
    print("\nDescribe:\n", df.describe())

    # Note: The groupby, merge, and one_hot methods are monkey-patched in __init__.py.
    # To use them here without running __init__.py, you might need to
    # import them and attach them to the DataFrame class explicitly.
    # For a simple demo, we'll assume they've been attached.
    
    # You will need to import the merge methods and groupby methods to run them.
    # Here's an example:
    from monkeyframe.groupby import _groupby_method
    DataFrame.groupby = _groupby_method
    print("\nGroupby city mean:\n", df.groupby("city", agg="mean"))
    
    from monkeyframe.encoders import _one_hot_method
    DataFrame.one_hot = _one_hot_method
    df2 = df.one_hot("city", drop_first=True)
    print("\nOne-hot (drop_first):_one_hot_method\n", df2)

    train, test = train_test_split(df2, test_size=0.25, stratify="y", random_state=42)
    print("\nTrain:\n", train)
    print("\nTest:\n", test)
    
    from monkeyframe.merge import _merge_method
    DataFrame.merge = _merge_method
    left = DataFrame({
        "city": np.array(["A", "B", "C"]),
        "x1": np.array([1, 2, 3])
    })
    right = DataFrame({
        "city": np.array(["A", "B"]),
        "z": np.array([100, 200])
    })
    print("Inner:\n", left.merge(right, on="city", how="inner"))
    print("\nLeft:\n", left.merge(right, on="city", how="left"))
    print("\nRight:\n", left.merge(right, on="city", how="right"))
    print("\nOuter:\n", left.merge(right, on="city", how="outer"))
