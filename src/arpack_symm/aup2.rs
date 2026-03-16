use crate::arpack_symm::aitr::dsaitr_mode1;
use crate::arpack_symm::apps::dsapps;
use crate::arpack_symm::conv::dsconv;
use crate::arpack_symm::sort::{dsgets, dsgets_with_order, dsortr};
use crate::arpack_symm::tridiag::dseigt_prefix;
use crate::{RuPcaError, Result};
use nalgebra::DMatrix;

pub struct Dsaup2Result {
    pub ritz: Vec<f64>,
    pub bounds: Vec<f64>,
    pub nconv: usize,
    pub nev: usize,
    pub np: usize,
    pub rnorm: f64,
    pub info: i32,
    pub basis_dim: usize,
    pub selected_indices: Vec<usize>,
}

pub fn dsaup2_mode1<F>(
    which: &str,
    nev0: usize,
    np0: usize,
    tol: f64,
    mxiter: usize,
    resid: &mut [f64],
    rnorm: &mut f64,
    v: &mut DMatrix<f64>,
    h: &mut DMatrix<f64>,
    mut op: F,
) -> Result<Dsaup2Result>
where
    F: FnMut(&[f64], &mut [f64]),
{
    if which == "BE" {
        return Err(RuPcaError::InvalidInput(
            "BE ordering not yet supported in dsaup2_mode1".to_string(),
        ));
    }
    let n = v.nrows();
    let kplusp = nev0 + np0;
    if v.ncols() < kplusp || h.nrows() < kplusp || h.ncols() != 2 {
        return Err(RuPcaError::InvalidInput(
            "V/H dimensions are inconsistent with dsaup2_mode1".to_string(),
        ));
    }

    let mut nev = nev0;
    let mut np = np0;
    let mut info = 0i32;
    let eps23 = f64::EPSILON.powf(2.0 / 3.0);

    let init = dsaitr_mode1(n, 0, nev0, resid, rnorm, v, h, |x, y| op(x, y))?;
    if init.info > 0 {
        return Ok(Dsaup2Result {
            ritz: Vec::new(),
            bounds: Vec::new(),
            nconv: 0,
            nev,
            np: init.info,
            rnorm: *rnorm,
            info: -9999,
            basis_dim: init.info,
            selected_indices: Vec::new(),
        });
    }

    let mut iter = 0usize;
    loop {
        iter += 1;
        let upd = dsaitr_mode1(n, nev, np, resid, rnorm, v, h, |x, y| op(x, y))?;
        if upd.info > 0 {
            return Ok(Dsaup2Result {
                ritz: Vec::new(),
                bounds: Vec::new(),
                nconv: 0,
                nev,
                np: upd.info,
                rnorm: *rnorm,
                info: -9999,
                basis_dim: upd.info,
                selected_indices: Vec::new(),
            });
        }

        let (mut ritz, mut bounds) = dseigt_prefix(*rnorm, h, kplusp);
        let mut order = (0..kplusp).collect::<Vec<_>>();

        nev = nev0;
        np = np0;
        let mut shifts = vec![0.0; np0];
        dsgets_with_order(1, which, nev, np, &mut ritz, &mut bounds, &mut order, &mut shifts);

        let nconv = dsconv(&ritz[np..np + nev], &bounds[np..np + nev], tol);

        let nptemp = np;
        for j in 0..nptemp {
            if bounds[j] == 0.0 {
                np -= 1;
                nev += 1;
            }
        }

        if nconv >= nev0 || iter > mxiter || np == 0 {
            let wprime = match which {
                "LM" => "SM",
                "SM" => "LM",
                "LA" => "SA",
                "SA" => "LA",
                _ => {
                    return Err(RuPcaError::InvalidInput(format!(
                        "unsupported WHICH={which}"
                    )))
                }
            };
            dsortr(wprime, true, &mut ritz, &mut bounds);

            for j in 0..nev0 {
                let temp = eps23.max(ritz[j].abs());
                bounds[j] /= temp;
            }
            dsortr("LA", true, &mut bounds[..nev0], &mut ritz[..nev0]);
            for j in 0..nev0 {
                let temp = eps23.max(ritz[j].abs());
                bounds[j] *= temp;
            }
            dsortr(which, true, &mut ritz[..nconv], &mut bounds[..nconv]);
            h[(0, 0)] = *rnorm;

            if iter > mxiter && nconv < nev {
                info = 1;
            }
            if np == 0 && nconv < nev0 {
                info = 2;
            }

            return Ok(Dsaup2Result {
                ritz,
                bounds,
                nconv,
                nev: nconv,
                np: nconv,
                rnorm: *rnorm,
                info,
                basis_dim: kplusp,
                selected_indices: order[np..np + nconv].to_vec(),
            });
        } else if nconv < nev {
            let nevbef = nev;
            nev += nconv.min(np / 2);
            if nev == 1 && kplusp >= 6 {
                nev = kplusp / 2;
            } else if nev == 1 && kplusp > 2 {
                nev = 2;
            }
            np = kplusp - nev;
            if nevbef < nev {
                shifts.resize(np, 0.0);
                dsgets(1, which, nev, np, &mut ritz, &mut bounds, &mut shifts);
            }
        }

        dsapps(nev, np, &ritz[..np], v, h, resid);
        *rnorm = resid.iter().map(|x| x * x).sum::<f64>().sqrt();
    }
}
