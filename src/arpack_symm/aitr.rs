use crate::{dot, RuPcaError, Result};
use nalgebra::DMatrix;

pub struct DsaitrResult {
    pub info: usize,
    pub rnorm: f64,
}

pub fn dsaitr_mode1<F>(
    n: usize,
    k: usize,
    np: usize,
    resid: &mut [f64],
    rnorm: &mut f64,
    v: &mut DMatrix<f64>,
    h: &mut DMatrix<f64>,
    mut op: F,
) -> Result<DsaitrResult>
where
    F: FnMut(&[f64], &mut [f64]),
{
    if v.nrows() != n || v.ncols() < k + np {
        return Err(RuPcaError::InvalidInput(
            "V dimensions are inconsistent with dsaitr arguments".to_string(),
        ));
    }
    if h.nrows() < k + np || h.ncols() != 2 {
        return Err(RuPcaError::InvalidInput(
            "H dimensions are inconsistent with dsaitr arguments".to_string(),
        ));
    }
    if resid.len() != n {
        return Err(RuPcaError::InvalidInput(
            "residual length mismatch".to_string(),
        ));
    }

    let safmin = f64::MIN_POSITIVE;
    let mut work_irj = vec![0.0; n];
    let mut work_ivj = vec![0.0; n];
    let mut fourier = vec![0.0; k + np];
    let mut corr = vec![0.0; k + np];

    let mut j = k;
    while j < k + np {
        if *rnorm <= 0.0 {
            return Ok(DsaitrResult {
                info: j,
                rnorm: *rnorm,
            });
        }

        let scale = if *rnorm >= safmin {
            1.0 / *rnorm
        } else {
            1.0 / (*rnorm).max(safmin)
        };
        {
            let mut vj = v.column_mut(j);
            for (row, dst) in vj.iter_mut().enumerate() {
                let value = resid[row] * scale;
                *dst = value;
                work_ivj[row] = value;
            }
        }
        op(&work_ivj, &mut work_irj);
        resid.copy_from_slice(&work_irj);

        let wnorm = dot(resid, resid).sqrt();

        for col in 0..=j {
            fourier[col] = dot(v.column(col).as_slice(), resid);
        }
        for col in 0..=j {
            let coeff = fourier[col];
            let vcol = v.column(col);
            let vcol_slice = vcol.as_slice();
            for row in 0..n {
                resid[row] -= coeff * vcol_slice[row];
            }
        }

        h[(j, 1)] = fourier[j];
        h[(j, 0)] = if j == 0 { 0.0 } else { *rnorm };
        *rnorm = dot(resid, resid).sqrt();

        let mut iter = 0usize;
        while *rnorm <= 0.717 * wnorm {
            for col in 0..=j {
                corr[col] = dot(v.column(col).as_slice(), resid);
            }
            for col in 0..=j {
                let coeff = corr[col];
                let vcol = v.column(col);
                let vcol_slice = vcol.as_slice();
                for row in 0..n {
                    resid[row] -= coeff * vcol_slice[row];
                }
            }
            h[(j, 1)] += corr[j];
            let rnorm1 = dot(resid, resid).sqrt();
            if rnorm1 > 0.717 * *rnorm {
                *rnorm = rnorm1;
                break;
            }
            *rnorm = rnorm1;
            iter += 1;
            if iter > 1 {
                for r in resid.iter_mut() {
                    *r = 0.0;
                }
                *rnorm = 0.0;
                break;
            }
        }

        if h[(j, 0)] < 0.0 {
            h[(j, 0)] = -h[(j, 0)];
            if j + 1 < k + np {
                for row in 0..n {
                    v[(row, j + 1)] = -v[(row, j + 1)];
                }
            } else {
                for r in resid.iter_mut() {
                    *r = -*r;
                }
            }
        }

        j += 1;
    }

    Ok(DsaitrResult {
        info: 0,
        rnorm: *rnorm,
    })
}
