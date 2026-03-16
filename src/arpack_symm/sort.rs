use nalgebra::DMatrix;

fn less(which: &str, a: f64, b: f64) -> bool {
    match which {
        "SA" => a < b,
        "SM" => a.abs() < b.abs(),
        "LA" => a > b,
        "LM" => a.abs() > b.abs(),
        _ => false,
    }
}

fn dsortr_by<T: Copy>(which: &str, x1: &mut [f64], x2: &mut [T]) {
    let n = x1.len();
    assert_eq!(x2.len(), n);
    let mut igap = n / 2;
    while igap > 0 {
        for i in igap..n {
            let mut j = i as isize - igap as isize;
            while j >= 0 {
                let jj = j as usize;
                if less(which, x1[jj], x1[jj + igap]) {
                    x1.swap(jj, jj + igap);
                    x2.swap(jj, jj + igap);
                } else {
                    break;
                }
                j -= igap as isize;
            }
        }
        igap /= 2;
    }
}

pub fn dsortr(which: &str, apply: bool, x1: &mut [f64], x2: &mut [f64]) {
    let n = x1.len();
    if apply {
        assert_eq!(x2.len(), n);
    }
    let mut igap = n / 2;
    while igap > 0 {
        for i in igap..n {
            let mut j = i as isize - igap as isize;
            while j >= 0 {
                let jj = j as usize;
                if less(which, x1[jj], x1[jj + igap]) {
                    x1.swap(jj, jj + igap);
                    if apply {
                        x2.swap(jj, jj + igap);
                    }
                } else {
                    break;
                }
                j -= igap as isize;
            }
        }
        igap /= 2;
    }
}

pub fn dsortr_usize(which: &str, x1: &mut [f64], x2: &mut [usize]) {
    dsortr_by(which, x1, x2);
}

pub fn dsesrt(which: &str, apply: bool, x: &mut [f64], a: &mut DMatrix<f64>) {
    let n = x.len();
    assert_eq!(a.ncols(), n);
    let mut igap = n / 2;
    while igap > 0 {
        for i in igap..n {
            let mut j = i as isize - igap as isize;
            while j >= 0 {
                let jj = j as usize;
                if less(which, x[jj], x[jj + igap]) {
                    x.swap(jj, jj + igap);
                    if apply {
                        a.swap_columns(jj, jj + igap);
                    }
                } else {
                    break;
                }
                j -= igap as isize;
            }
        }
        igap /= 2;
    }
}

pub fn dsgets(
    ishift: i32,
    which: &str,
    kev: usize,
    np: usize,
    ritz: &mut [f64],
    bounds: &mut [f64],
    shifts: &mut [f64],
) {
    assert_eq!(ritz.len(), kev + np);
    assert_eq!(bounds.len(), kev + np);
    if which == "BE" {
        dsortr("LA", true, ritz, bounds);
        let kevd2 = kev / 2;
        if kev > 1 {
            let nswap = kevd2.min(np);
            let offset = kevd2.max(np);
            for i in 0..nswap {
                ritz.swap(i, offset + i);
                bounds.swap(i, offset + i);
            }
        }
    } else {
        dsortr(which, true, ritz, bounds);
    }

    if ishift == 1 && np > 0 {
        let (b_unwanted, r_unwanted) = (&mut bounds[..np], &mut ritz[..np]);
        dsortr("SM", true, b_unwanted, r_unwanted);
        shifts[..np].copy_from_slice(r_unwanted);
    }
}

pub fn dsgets_with_order(
    ishift: i32,
    which: &str,
    kev: usize,
    np: usize,
    ritz: &mut [f64],
    bounds: &mut [f64],
    order: &mut [usize],
    shifts: &mut [f64],
) {
    assert_eq!(ritz.len(), kev + np);
    assert_eq!(bounds.len(), kev + np);
    assert_eq!(order.len(), kev + np);
    if which == "BE" {
        dsortr_by("LA", ritz, order);
        let mut reordered_bounds = vec![0.0; bounds.len()];
        for (new_pos, &old_pos) in order.iter().enumerate() {
            reordered_bounds[new_pos] = bounds[old_pos];
        }
        bounds.copy_from_slice(&reordered_bounds);
        let kevd2 = kev / 2;
        if kev > 1 {
            let nswap = kevd2.min(np);
            let offset = kevd2.max(np);
            for i in 0..nswap {
                ritz.swap(i, offset + i);
                bounds.swap(i, offset + i);
                order.swap(i, offset + i);
            }
        }
    } else {
        dsortr_by(which, ritz, order);
        let mut reordered_bounds = vec![0.0; bounds.len()];
        for (new_pos, &old_pos) in order.iter().enumerate() {
            reordered_bounds[new_pos] = bounds[old_pos];
        }
        bounds.copy_from_slice(&reordered_bounds);
    }

    if ishift == 1 && np > 0 {
        let (b_unwanted, r_unwanted) = (&mut bounds[..np], &mut ritz[..np]);
        dsortr("SM", true, b_unwanted, r_unwanted);
        shifts[..np].copy_from_slice(r_unwanted);
    }
}
