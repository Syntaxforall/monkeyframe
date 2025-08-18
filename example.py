import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as skl_train_test_split

# Try importing polars if available
try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

from monkeyframe.dataframe import DataFrame
from monkeyframe.ml import train_test_split as ff_train_test_split, shuffle_df as ff_shuffle_df

# --- Setup synthetic dataset ---
N = 5_000_000  # adjust to your system's RAM
np.random.seed(42)

cities = np.random.choice(["A", "B", "C", "D", "E"], size=N).astype(object)
x1 = np.random.rand(N) * 100
x2 = np.random.randint(0, 1000, size=N)
y = np.random.randint(0, 2, size=N)

# Pandas DataFrame
pdf = pd.DataFrame({"city": cities, "x1": x1, "x2": x2, "y": y})

# monkeyframe DataFrame
ffdf = DataFrame({"city": cities, "x1": x1, "x2": x2, "y": y})

# Polars DataFrame (optional)
if POLARS_AVAILABLE:
    pldf = pl.DataFrame({"city": cities, "x1": x1, "x2": x2, "y": y})

# --- Benchmark helper ---
def benchmark(name, func, repeat=3):
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append(end - start)
    avg_time = sum(times) / repeat
    return (name, avg_time)

# --- Run benchmarks ---
results = []
# dataframe 
results.append(benchmark("monkeyframe DataFrame creation", lambda: DataFrame({"city": cities, "x1": x1, "x2": x2, "y": y})))
results.append(benchmark("Pandas DataFrame creation", lambda: pd.DataFrame({"city": cities, "x1": x1, "x2": x2, "y": y})))
if POLARS_AVAILABLE:
    results.append(benchmark("Polars DataFrame creation", lambda: pl.DataFrame({"city": cities, "x1": x1, "x2": x2, "y": y})))

# 1. GroupBy mean
results.append(benchmark("monkeyframe groupby mean", lambda: ffdf.groupby("city").mean()))
results.append(benchmark("Pandas groupby mean", lambda: pdf.groupby("city")[["x1", "x2"]].mean()))
if POLARS_AVAILABLE:
    results.append(benchmark("Polars groupby mean", lambda: pldf.group_by("city").agg(pl.col("x1").mean(), pl.col("x2").mean())))

# 2. Filtering
results.append(benchmark("monkeyframe filter", lambda: ffdf.filter((ffdf["x1"] > 50) & (ffdf["y"] == 1))))
results.append(benchmark("Pandas filter", lambda: pdf[(pdf["x1"] > 50) & (pdf["y"] == 1)]))
if POLARS_AVAILABLE:
    results.append(benchmark("Polars filter", lambda: pldf.filter((pl.col("x1") > 50) & (pl.col("y") == 1))))

# 3. Sort
results.append(benchmark("monkeyframe sort", lambda: ffdf.sort_values("x2", ascending=False)))
results.append(benchmark("Pandas sort", lambda: pdf.sort_values("x2", ascending=False)))
if POLARS_AVAILABLE:
    results.append(benchmark("Polars sort", lambda: pldf.sort("x2", descending=True)))

# 4. Standardize
numeric_cols = ["x1", "x2"]
results.append(benchmark("monkeyframe standardize", lambda: ffdf.standardize(cols=numeric_cols)))
results.append(benchmark("Pandas standardize", lambda: (pdf[numeric_cols] - pdf[numeric_cols].mean()) / pdf[numeric_cols].std()))
if POLARS_AVAILABLE:
    results.append(benchmark("Polars standardize", lambda: pldf.select([(pl.col(c) - pl.col(c).mean()) / pl.col(c).std() for c in numeric_cols])))

# 5. Merge
cities_unique = ["A", "B", "C", "D", "E"]
extra_data_dict = {
    'city': np.array(cities_unique),
    'country': np.array(['USA', 'UK', 'Canada', 'Germany', 'France'])
}
extra_pdf = pd.DataFrame(extra_data_dict)
extra_ffdf = DataFrame(extra_data_dict)
if POLARS_AVAILABLE:
    extra_pldf = pl.DataFrame(extra_data_dict)

results.append(benchmark("monkeyframe merge", lambda: ffdf.merge(extra_ffdf, on='city', how='left')))
results.append(benchmark("Pandas merge", lambda: pdf.merge(extra_pdf, on='city', how='left')))
if POLARS_AVAILABLE:
    results.append(benchmark("Polars merge", lambda: pldf.join(extra_pldf, on='city', how='left')))

# 6. Label Encode
results.append(benchmark("monkeyframe label_encode", lambda: ffdf.label_encode('city')))
results.append(benchmark("Pandas label_encode", lambda: pdf['city'].astype('category').cat.codes))
if POLARS_AVAILABLE:
    results.append(benchmark("Polars label_encode", lambda: pldf.with_columns(pl.col('city').cast(pl.Categorical).to_physical())))

# 7. One-Hot Encode
results.append(benchmark("monkeyframe one_hot", lambda: ffdf.one_hot('city')))
results.append(benchmark("Pandas one_hot", lambda: pd.get_dummies(pdf, columns=['city'])))
if POLARS_AVAILABLE:
    results.append(benchmark("Polars one_hot", lambda: pldf.to_dummies(columns=['city'])))

# 8. Shuffle
results.append(benchmark("monkeyframe shuffle", lambda: ff_shuffle_df(ffdf)))
results.append(benchmark("Pandas shuffle", lambda: pdf.sample(frac=1)))
if POLARS_AVAILABLE:
    results.append(benchmark("Polars shuffle", lambda: pldf.sample(fraction=1.0, shuffle=True)))

# 9. Train-Test Split
results.append(benchmark("monkeyframe train_test_split", lambda: ff_train_test_split(ffdf, target='y', test_size=0.2)))
results.append(benchmark("Sklearn train_test_split", lambda: skl_train_test_split(pdf.drop('y', axis=1), pdf['y'], test_size=0.2)))


# --- Show results ---
results_df = pd.DataFrame(results, columns=["Operation", "Avg Time (s)"])
print("\nBenchmark Results:")
print(results_df.to_string(index=False))
