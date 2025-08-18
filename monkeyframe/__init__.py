"""
monkeyframe: A minimal, fast DataFrame library for ML workflows.
"""
from .dataframe import DataFrame
from .series import Series
from .index import Index
from .groupby import groupby
from .merge import merge
from .encoders import label_encode, one_hot, standardize
DataFrame.groupby = groupby
DataFrame.merge = merge
DataFrame.label_encode = label_encode
from .ml import train_test_split, kfold, shuffle_df
DataFrame.one_hot = one_hot
DataFrame.standardize = standardize
DataFrame.train_test_split = train_test_split
DataFrame.shuffle = shuffle_df