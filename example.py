import time
import numpy as np
import pandas as pd

# Try importing polars if available
try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

from monkeyframe.dataframe import DataFrame
from monkeyframe.groupby import _groupby_method
DataFrame.groupby = _groupby_method  # monkey-patch

# --- Setup synthetic dataset ---
N = 5_000_000  # adjust to your system's RAM
np.random.seed(42)

cities = np.random.choice(["A", "B", "C", "D", "E"], size=N)
x1 = np.random.rand(N) * 100
x2 = np.random.randint(0, 1000, size=N)
y = np.random.randint(0, 2, size=N)

# Pandas DataFrame
pdf = pd.DataFrame({"city": cities, "x1": x1, "x2": x2, "y": y})

# monkeyframe DataFrame
ffdf = DataFrame({"city": cities, "x1": x1, "x2": x2, "y": y})

# Polars DataFrame (optional)

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
results.append(benchmark("monkeyframe groupby mean", lambda: ffdf.groupby("city", agg="mean")))
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

# --- Show results ---
results_df = pd.DataFrame(results, columns=["Operation", "Avg Time (s)"])
print("\nBenchmark Results:")
print(results_df.to_string(index=False))
