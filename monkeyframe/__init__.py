"""
monkeyframe: A minimal, fast DataFrame library for ML workflows.
"""
from .dataframe import DataFrame
from .groupby import _groupby_method
from .merge import _merge_method
from .encoders import _one_hot_method, _label_encode_method, _standardize_method
from .ml import train_test_split, kfold, shuffle_df

# Monkey-patch all methods onto the DataFrame class
DataFrame.groupby = _groupby_method
DataFrame.merge = _merge_method
DataFrame.one_hot = _one_hot_method
DataFrame.label_encode = _label_encode_method
DataFrame.standardize = _standardize_method