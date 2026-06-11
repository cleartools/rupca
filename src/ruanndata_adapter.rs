use crate::{
    pca_scanpy_dense, pca_scanpy_sparse_csr, pca_shifted_clr_sparse_csr, CsrMatrix, DenseMatrix,
    Result, RuPcaError, ScanpyPcaParams, ScanpyPcaResult, ShiftedClrCsrMatrix,
};
use ruanndata::{ArrayData, ArrayValue, MatrixData};

pub fn pca_scanpy_ruanndata(x: &MatrixData, params: ScanpyPcaParams) -> Result<ScanpyPcaResult> {
    match x {
        MatrixData::Dense { .. } => pca_scanpy_dense(&dense_from_ruanndata(x)?, params),
        MatrixData::Csr { .. } | MatrixData::Csc { .. } => {
            pca_scanpy_sparse_csr(&sparse_csr_from_ruanndata(x)?, params)
        }
        MatrixData::ShiftedClrCsr { .. } => {
            pca_shifted_clr_sparse_csr(&shifted_clr_csr_from_ruanndata(x)?, params)
        }
    }
}

pub fn sparse_csr_from_ruanndata(x: &MatrixData) -> Result<CsrMatrix> {
    match x {
        MatrixData::Csr {
            n_rows,
            n_cols,
            data,
            indices,
            indptr,
        } => {
            let csr = CsrMatrix {
                n_rows: *n_rows,
                n_cols: *n_cols,
                data: values_as_f64(data)?,
                indices: indices.clone(),
                indptr: indptr.clone(),
            };
            csr.validate()?;
            Ok(csr)
        }
        MatrixData::Csc {
            n_rows,
            n_cols,
            data,
            indices,
            indptr,
        } => csc_to_csr(*n_rows, *n_cols, data, indices, indptr),
        MatrixData::Dense { .. } => Err(RuPcaError::InvalidInput(
            "ruanndata dense matrix should be passed through dense_from_ruanndata".to_string(),
        )),
        MatrixData::ShiftedClrCsr { .. } => Err(RuPcaError::InvalidInput(
            "ruanndata shifted-CLR matrix should be passed through shifted_clr_csr_from_ruanndata"
                .to_string(),
        )),
    }
}

pub fn dense_from_ruanndata(x: &MatrixData) -> Result<DenseMatrix> {
    match x {
        MatrixData::Dense { array } => {
            let shape = array.shape();
            if shape.len() != 2 {
                return Err(RuPcaError::InvalidInput(format!(
                    "ruanndata dense matrix must be rank 2, found shape {:?}",
                    shape
                )));
            }
            let dense = DenseMatrix {
                n_rows: shape[0],
                n_cols: shape[1],
                data: values_as_f64(array)?,
            };
            dense.validate()?;
            Ok(dense)
        }
        _ => Err(RuPcaError::InvalidInput(
            "expected ruanndata MatrixData::Dense".to_string(),
        )),
    }
}

pub fn shifted_clr_csr_from_ruanndata(x: &MatrixData) -> Result<ShiftedClrCsrMatrix> {
    match x {
        MatrixData::ShiftedClrCsr {
            n_rows,
            n_cols,
            data,
            indices,
            indptr,
            row_center,
        } => {
            let shifted = ShiftedClrCsrMatrix {
                sparse: CsrMatrix {
                    n_rows: *n_rows,
                    n_cols: *n_cols,
                    data: values_as_f64(data)?,
                    indices: indices.clone(),
                    indptr: indptr.clone(),
                },
                row_center: values_as_f64(row_center)?,
            };
            shifted.validate()?;
            Ok(shifted)
        }
        _ => Err(RuPcaError::InvalidInput(
            "expected ruanndata MatrixData::ShiftedClrCsr".to_string(),
        )),
    }
}

fn csc_to_csr(
    n_rows: usize,
    n_cols: usize,
    data: &ArrayData,
    indices: &[usize],
    indptr: &[usize],
) -> Result<CsrMatrix> {
    let values = values_as_f64(data)?;
    if indices.len() != values.len() {
        return Err(RuPcaError::InvalidInput(
            "CSC data and indices length mismatch".to_string(),
        ));
    }
    if indptr.len() != n_cols + 1 {
        return Err(RuPcaError::InvalidInput(
            "CSC indptr length must be n_cols + 1".to_string(),
        ));
    }
    if *indptr.first().unwrap_or(&0) != 0 {
        return Err(RuPcaError::InvalidInput(
            "CSC indptr must start at 0".to_string(),
        ));
    }
    if *indptr.last().unwrap_or(&0) != values.len() {
        return Err(RuPcaError::InvalidInput(
            "CSC indptr must end at nnz".to_string(),
        ));
    }
    for window in indptr.windows(2) {
        if window[0] > window[1] {
            return Err(RuPcaError::InvalidInput(
                "CSC indptr must be nondecreasing".to_string(),
            ));
        }
    }
    if indices.iter().any(|&i| i >= n_rows) {
        return Err(RuPcaError::InvalidInput(
            "CSC row index out of bounds".to_string(),
        ));
    }

    let mut row_counts = vec![0usize; n_rows];
    for &row in indices {
        row_counts[row] += 1;
    }

    let mut out_indptr = vec![0usize; n_rows + 1];
    for row in 0..n_rows {
        out_indptr[row + 1] = out_indptr[row] + row_counts[row];
    }

    let mut next = out_indptr.clone();
    let mut out_indices = vec![0usize; values.len()];
    let mut out_data = vec![0.0; values.len()];
    for col in 0..n_cols {
        for offset in indptr[col]..indptr[col + 1] {
            let row = indices[offset];
            let dest = next[row];
            out_indices[dest] = col;
            out_data[dest] = values[offset];
            next[row] += 1;
        }
    }

    let csr = CsrMatrix {
        n_rows,
        n_cols,
        data: out_data,
        indices: out_indices,
        indptr: out_indptr,
    };
    csr.validate()?;
    Ok(csr)
}

fn values_as_f64(array: &ArrayData) -> Result<Vec<f64>> {
    if array.len() != array.values.len() {
        return Err(RuPcaError::InvalidInput(format!(
            "ruanndata array stores {} values but shape {:?} requires {}",
            array.values.len(),
            array.shape(),
            array.len()
        )));
    }
    Ok(match &array.values {
        ArrayValue::Float64(v) => v.clone(),
        ArrayValue::Float32(v) => v.iter().map(|x| *x as f64).collect(),
        ArrayValue::Int64(v) => v.iter().map(|x| *x as f64).collect(),
        ArrayValue::Int32(v) => v.iter().map(|x| *x as f64).collect(),
        ArrayValue::UInt64(v) => v.iter().map(|x| *x as f64).collect(),
        ArrayValue::UInt32(v) => v.iter().map(|x| *x as f64).collect(),
        ArrayValue::Bool(v) => v.iter().map(|x| if *x { 1.0 } else { 0.0 }).collect(),
        ArrayValue::String(_) => {
            return Err(RuPcaError::InvalidInput(
                "string arrays cannot be converted to rupca matrices".to_string(),
            ))
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> ScanpyPcaParams {
        ScanpyPcaParams {
            n_components: 2,
            ..ScanpyPcaParams::default()
        }
    }

    #[test]
    fn runs_on_ruanndata_csr() {
        let matrix = MatrixData::Csr {
            n_rows: 5,
            n_cols: 3,
            data: ArrayData {
                shape: vec![7],
                values: ArrayValue::Float64(vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 5.0]),
            },
            indices: vec![0, 2, 1, 0, 2, 1, 2],
            indptr: vec![0, 2, 3, 5, 6, 7],
        };

        let got = pca_scanpy_ruanndata(&matrix, params()).unwrap();
        assert_eq!(got.n_components, 2);
        assert!(got.warnings.is_empty());
    }

    #[test]
    fn runs_on_ruanndata_dense() {
        let matrix = MatrixData::Dense {
            array: ArrayData {
                shape: vec![4, 3],
                values: ArrayValue::Float64(vec![
                    1.0, 0.0, 2.0, 0.0, 3.0, 1.0, 2.0, 1.0, 0.0, 4.0, 0.0, 1.0,
                ]),
            },
        };

        let got = pca_scanpy_ruanndata(&matrix, params()).unwrap();
        assert_eq!(got.n_components, 2);
        assert_eq!(got.warnings.len(), 1);
    }

    #[test]
    fn runs_on_ruanndata_shifted_clr_csr() {
        let matrix = MatrixData::ShiftedClrCsr {
            n_rows: 5,
            n_cols: 3,
            data: ArrayData {
                shape: vec![7],
                values: ArrayValue::Float64(vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 5.0]),
            },
            indices: vec![0, 2, 1, 0, 2, 1, 2],
            indptr: vec![0, 2, 3, 5, 6, 7],
            row_center: ArrayData {
                shape: vec![5],
                values: ArrayValue::Float64(vec![1.0, 1.0, 5.0 / 3.0, 2.0 / 3.0, 5.0 / 3.0]),
            },
        };

        let got = pca_scanpy_ruanndata(&matrix, params()).unwrap();
        assert_eq!(got.n_components, 2);
        assert!(got.warnings.is_empty());
    }

    #[test]
    fn converts_ruanndata_csc_to_csr() {
        let matrix = MatrixData::Csc {
            n_rows: 3,
            n_cols: 3,
            data: ArrayData {
                shape: vec![4],
                values: ArrayValue::Float64(vec![1.0, 4.0, 3.0, 2.0]),
            },
            indices: vec![0, 2, 1, 0],
            indptr: vec![0, 2, 3, 4],
        };

        let got = sparse_csr_from_ruanndata(&matrix).unwrap();
        assert_eq!(got.n_rows, 3);
        assert_eq!(got.n_cols, 3);
        assert_eq!(got.indptr, vec![0, 2, 3, 4]);
        assert_eq!(got.indices, vec![0, 2, 1, 0]);
        assert_eq!(got.data, vec![1.0, 2.0, 3.0, 4.0]);
    }
}
