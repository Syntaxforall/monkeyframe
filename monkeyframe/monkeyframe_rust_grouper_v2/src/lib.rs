use pyo3::prelude::*;
use pyo3::types::PyList;
use arrow::pyarrow::{FromPyArrow, ToPyArrow};
use arrow::array::{Array, StringArray};
use std::collections::HashMap;

#[pyfunction]
fn hash_grouper_rust(key_array: &PyAny) -> PyResult<(Vec<i64>, PyObject)> {
    let key_array = StringArray::from_pyarrow(key_array)?;

    let mut group_map = HashMap::new();
    let mut group_ids = Vec::with_capacity(key_array.len());
    let mut unique_keys: Vec<&str> = Vec::new();
    let mut next_group_id: i64 = 0;

    for key in key_array.iter() {
        if let Some(key) = key {
            let entry = group_map.entry(key).or_insert_with(|| {
                let id = next_group_id;
                unique_keys.push(key);
                next_group_id += 1;
                id
            });
            group_ids.push(*entry);
        } else {
            group_ids.push(-1); // Null group
        }
    }

    Python::with_gil(|py| {
        let unique_keys_list = PyList::new(py, &unique_keys);
        Ok((group_ids, unique_keys_list.into()))
    })
}

#[pymodule]
fn monkeyframe_rust_grouper_v2(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hash_grouper_rust, m)?)?;
    Ok(())
}
