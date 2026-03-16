use nalgebra::{DMatrix, SymmetricEigen};

pub fn dseigt_prefix(rnorm: f64, h: &DMatrix<f64>, n: usize) -> (Vec<f64>, Vec<f64>) {
    assert!(h.nrows() >= n);
    assert_eq!(h.ncols(), 2);
    let mut t = DMatrix::zeros(n, n);
    for i in 0..n {
        t[(i, i)] = h[(i, 1)];
        if i > 0 {
            t[(i, i - 1)] = h[(i, 0)];
            t[(i - 1, i)] = h[(i, 0)];
        }
    }
    let eig = SymmetricEigen::new(t);
    let mut order = (0..n).collect::<Vec<_>>();
    order.sort_by(|&a, &b| {
        eig.eigenvalues[a]
            .partial_cmp(&eig.eigenvalues[b])
            .unwrap()
    });
    let mut eigvals = Vec::with_capacity(n);
    let mut bounds = Vec::with_capacity(n);
    for idx in order {
        eigvals.push(eig.eigenvalues[idx]);
        bounds.push(rnorm * eig.eigenvectors[(n - 1, idx)].abs());
    }
    (eigvals, bounds)
}

pub fn dseigt(rnorm: f64, h: &DMatrix<f64>) -> (Vec<f64>, Vec<f64>) {
    dseigt_prefix(rnorm, h, h.nrows())
}
