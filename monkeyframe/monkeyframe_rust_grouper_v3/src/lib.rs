use pyo3::prelude::*;
use pyo3::types::PyList;
use numpy::{PyArray1, ToPyArray};
use arrow::pyarrow::FromPyArrow;
use arrow::array::{Array, StringArray};
use ahash::AHashMap;
use rayon::prelude::*;

#[pyfunction]
fn hash_grouper_rust_v3<'py>(py: Python<'py>, key_array: &PyAny) -> PyResult<(Bound<'py, PyArray1<i64>>, PyObject)> {
    let key_array = StringArray::from_pyarrow(key_array)?;

    // Build the map sequentially
    let mut group_map = AHashMap::new();
    let mut unique_keys: Vec<&str> = Vec::new();
    let mut next_group_id: i64 = 0;

    for key in key_array.iter() {
        if let Some(key) = key {
            group_map.entry(key).or_insert_with(|| {
                let id = next_group_id;
                unique_keys.push(key);
                next_group_id += 1;
                id
            });
        }
    }

    // Create the group_ids array in parallel
    let group_ids: Vec<i64> = (0..key_array.len()).into_par_iter().map(|i| {
        if let Some(key) = key_array.get(i) {
            *group_map.get(key).unwrap()
        } else {
            -1
        }
    }).collect();

    let group_ids_np = group_ids.to_pyarray_bound(py);
    let unique_keys_list = PyList::new_bound(py, &unique_keys);
    Ok((group_ids_np, unique_keys_list.into()))
}

#[pymodule]
fn monkeyframe_rust_grouper_v3(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hash_grouper_rust_v3, m)?)?;
    Ok(())
}
