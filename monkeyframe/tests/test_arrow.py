import numpy as np
import pyarrow as pa
import pytest
from monkeyframe.dataframe import DataFrame

@pytest.mark.skip(reason="Not implemented on new architecture yet")
def test_arrow_conversion():
    # 1. Create a monkeyframe DataFrame
    df = DataFrame({
        'A': np.array([1, 2, 3, 4]),
        'B': np.array([1.1, 2.2, 3.3, 4.4]),
        'C': np.array(['foo', 'bar', 'baz', 'qux'], dtype=object)
    })

    # 2. Convert to Arrow Table
    arrow_table = df.to_arrow()

    # Check that the result is a pyarrow.Table
    assert isinstance(arrow_table, pa.Table)

    # Check schema
    assert arrow_table.schema.names == ['A', 'B', 'C']
    assert arrow_table.schema.field('A').type == pa.int64()
    assert arrow_table.schema.field('B').type == pa.float64()
    assert arrow_table.schema.field('C').type == pa.string()

    # 3. Convert back to monkeyframe DataFrame
    df_from_arrow = DataFrame.from_arrow(arrow_table)

    # Check that the result is a monkeyframe.DataFrame
    assert isinstance(df_from_arrow, DataFrame)

    # Check that the data is the same
    assert df_from_arrow.columns == df.columns
    assert df_from_arrow.length == df.length
    for col in df.columns:
        assert np.array_equal(df_from_arrow[col], df[col])

@pytest.mark.skip(reason="Not implemented on new architecture yet")
def test_from_arrow_type_error():
    import pytest
    with pytest.raises(TypeError):
        DataFrame.from_arrow("not a table")
