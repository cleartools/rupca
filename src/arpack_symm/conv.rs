pub fn dsconv(ritz: &[f64], bounds: &[f64], tol: f64) -> usize {
    let eps23 = f64::EPSILON.powf(2.0 / 3.0);
    ritz.iter()
        .zip(bounds.iter())
        .filter(|(ritz_i, bound_i)| **bound_i <= tol * eps23.max(ritz_i.abs()))
        .count()
}
