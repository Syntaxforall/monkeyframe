# 🐒 MonkeyFrame

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)  
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)]()  
[![Build](https://img.shields.io/badge/build-passing-brightgreen)]()

**MonkeyFrame** is a fast, lightweight **DataFrame library** built with **NumPy + Numba**.  
Designed for **machine learning workflows** with built-in helpers for data preprocessing, train/test splitting, and K-fold cross-validation.

---

## 🚀 Features

### 📦 Core Data Structures
- `DataFrame` class → columnar storage using NumPy arrays
- Array-based operations for **high performance**

### 📂 Data Handling
- Create from dictionary: `{column_name: numpy_array}`
- Introspection: `shape`, `dtypes`, `to_numpy()`
- Indexing: `df['col']`, `df[['c1', 'c2']]`, boolean filtering
- Column operations: `assign()`, `drop()`
- View subsets: `head()`, `tail()`
- Sorting: `sort_values()`
- Missing data: `fillna()`
- Type conversion: `astype()`

### ➕ Arithmetic Operations
- Elementwise: `add()`, `sub()`, `mul()`, `div()`

### 📊 Descriptive Statistics
- `describe()` → count, mean, std, min, quartiles, max

### 🔄 GroupBy & Aggregation
- `groupby()` with **mean, sum, count** (Numba-accelerated)

### 🧠 Machine Learning Helpers
- `one_hot()` → one-hot encoding
- `label_encode()` → label encoding
- `standardize()` → z-score scaling
- `train_test_split()` → split data into train/test sets
- `kfold()` → K-fold cross-validation splits
- `shuffle_df()` → randomize rows

### 🔗 Merging & Joining
- SQL-style joins: `inner`, `left`, `right`, `outer`

---

## ⚡ Installation

Currently local install only:

```bash
git clone https://github.com/Syntaxforall/monkeyframe.git
cd monkeyframe
pip install -e .
```
Quick Example

```bash
import numpy as np
from monkeyframe import DataFrame

# Create a DataFrame
df = DataFrame({
    "age": np.array([23, 25, 31, 23, 40]),
    "salary": np.array([50000, 60000, 75000, 52000, 90000]),
    "dept": np.array(["IT", "HR", "IT", "Finance", "Finance"])
})

# View top rows
print(df.head())

# GroupBy + Mean
print(df.groupby("dept").agg("mean"))

# Train/test split for ML
train, test = df.train_test_split(test_size=0.2, shuffle=True)
```
📊 Benchmark (5M rows x 4 cols)

Pandas: ~4.5s

MonkeyFrame: ~1.6–1.8s
