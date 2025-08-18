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
from monkeyframe.series import Series
from monkeyframe.index import Index

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

if __name__ == "__main__":
    results = []

    # --- High-Cardinality GroupBy Benchmark ---
    N = 100_000
    n_unique_keys = 10_000
    np.random.seed(42)

    keys = np.random.randint(0, n_unique_keys, size=N)
    high_card_col = np.char.add("key_", keys.astype(str)).astype(object)
    numeric_col = np.random.rand(N)

    ffdf_hc = DataFrame({'key': high_card_col, 'val': numeric_col})
    pdf_hc = pd.DataFrame({'key': high_card_col, 'val': numeric_col})
    if POLARS_AVAILABLE:
        pldf_hc = pl.DataFrame({'key': high_card_col, 'val': numeric_col})

    results.append(benchmark("High-Card GroupBy (monkeyframe)", lambda: ffdf_hc.groupby('key').mean()))
    results.append(benchmark("High-Card GroupBy (pandas)", lambda: pdf_hc.groupby('key').mean()))
    if POLARS_AVAILABLE:
        results.append(benchmark("High-Card GroupBy (polars)", lambda: pldf_hc.group_by('key').agg(pl.col('val').mean())))

    # --- Wide DataFrame Benchmark ---
    N_wide = 100_000
    n_cols = 50
    np.random.seed(42)

    wide_data = {f"col_{i}": np.random.rand(N_wide) for i in range(n_cols)}

    ffdf_wide = DataFrame(wide_data)
    pdf_wide = pd.DataFrame(wide_data)
    if POLARS_AVAILABLE:
        pldf_wide = pl.DataFrame(wide_data)

    results.append(benchmark("Wide Creation (monkeyframe)", lambda: DataFrame(wide_data)))
    results.append(benchmark("Wide Creation (pandas)", lambda: pd.DataFrame(wide_data)))
    if POLARS_AVAILABLE:
        results.append(benchmark("Wide Creation (polars)", lambda: pl.DataFrame(wide_data)))

    results.append(benchmark("Wide Filter (monkeyframe)", lambda: ffdf_wide.filter(ffdf_wide['col_0'] > 0.5)))
    results.append(benchmark("Wide Filter (pandas)", lambda: pdf_wide[pdf_wide['col_0'] > 0.5]))
    if POLARS_AVAILABLE:
        results.append(benchmark("Wide Filter (polars)", lambda: pldf_wide.filter(pl.col('col_0') > 0.5)))

    results.append(benchmark("Wide Sort (monkeyframe)", lambda: ffdf_wide.sort_values('col_1')))
    results.append(benchmark("Wide Sort (pandas)", lambda: pdf_wide.sort_values('col_1')))
    if POLARS_AVAILABLE:
        results.append(benchmark("Wide Sort (polars)", lambda: pldf_wide.sort('col_1')))

    # --- value_counts Benchmark ---
    results.append(benchmark("value_counts (monkeyframe)", lambda: ffdf_hc['key'].value_counts()))
    results.append(benchmark("value_counts (pandas)", lambda: pdf_hc['key'].value_counts()))
    if POLARS_AVAILABLE:
        results.append(benchmark("value_counts (polars)", lambda: pldf_hc['key'].value_counts()))

    # --- Memory Usage ---
    ffdf_mem = ffdf_wide.memory_usage
    pdf_mem = pdf_wide.memory_usage(deep=True).sum()
    if POLARS_AVAILABLE:
        pldf_mem = pldf_wide.estimated_size()

    print("\n--- Memory Usage ---")
    print(f"monkeyframe: {ffdf_mem / 1e6:.2f} MB")
    print(f"pandas:      {pdf_mem / 1e6:.2f} MB")
    if POLARS_AVAILABLE:
        print(f"polars:      {pldf_mem / 1e6:.2f} MB")


    results_df = pd.DataFrame(results, columns=["Operation", "Avg Time (s)"])
    print("\nNew Benchmark Results:")
    print(results_df.to_string(index=False))
