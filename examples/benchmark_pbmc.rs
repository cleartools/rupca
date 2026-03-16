use rupca::{pca_scanpy_sparse_csr, CsrMatrix, ScanpyPcaParams};
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;

fn read_matrix_market(path: &str) -> CsrMatrix {
    let file = File::open(path).expect("failed to open matrix");
    let reader = BufReader::new(file);

    let mut n_rows = 0usize;
    let mut n_cols = 0usize;
    let mut nnz = 0usize;
    let mut triplets: Vec<(usize, usize, f64)> = Vec::new();
    let mut header_seen = false;

    for line in reader.lines() {
        let line = line.expect("failed to read line");
        if line.starts_with('%') || line.trim().is_empty() {
            continue;
        }
        if !header_seen {
            let parts = line.split_whitespace().collect::<Vec<_>>();
            n_rows = parts[0].parse().unwrap();
            n_cols = parts[1].parse().unwrap();
            nnz = parts[2].parse().unwrap();
            triplets.reserve(nnz);
            header_seen = true;
            continue;
        }
        let parts = line.split_whitespace().collect::<Vec<_>>();
        let i = parts[0].parse::<usize>().unwrap() - 1;
        let j = parts[1].parse::<usize>().unwrap() - 1;
        let v = parts[2].parse::<f64>().unwrap();
        triplets.push((i, j, v));
    }

    triplets.sort_unstable_by_key(|&(i, j, _)| (i, j));
    let mut data = Vec::with_capacity(nnz);
    let mut indices = Vec::with_capacity(nnz);
    let mut indptr = vec![0usize; n_rows + 1];
    let mut current_row = 0usize;
    for (i, j, v) in triplets {
        while current_row < i {
            current_row += 1;
            indptr[current_row] = data.len();
        }
        indices.push(j);
        data.push(v);
    }
    while current_row < n_rows {
        current_row += 1;
        indptr[current_row] = data.len();
    }

    CsrMatrix {
        n_rows,
        n_cols,
        data,
        indices,
        indptr,
    }
}

fn main() {
    let args = env::args().collect::<Vec<_>>();
    let matrix_path = args
        .get(1)
        .map(String::as_str)
        .unwrap_or("/Users/lpachter/Dropbox/claude/projects/rutest/pbmc10k/matrix.mtx");
    let n_components = args
        .get(2)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(50);
    let repeats = args
        .get(3)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(3);
    let ncv = args.get(4).and_then(|s| s.parse::<usize>().ok());

    let load_start = Instant::now();
    let matrix = read_matrix_market(matrix_path);
    let load_secs = load_start.elapsed().as_secs_f64();

    let mut best = f64::INFINITY;
    let mut sum = 0.0;
    for _ in 0..repeats {
        let start = Instant::now();
        let result = pca_scanpy_sparse_csr(
            &matrix,
            ScanpyPcaParams {
                n_components,
                tol: 0.0,
                ncv,
                maxiter: None,
                seed: 0,
            },
        )
        .expect("pca failed");
        let secs = start.elapsed().as_secs_f64();
        best = best.min(secs);
        sum += secs;
        std::hint::black_box(result);
    }

    println!(
        "{{\"rows\":{},\"cols\":{},\"nnz\":{},\"load_seconds\":{:.6},\"pca_best_seconds\":{:.6},\"pca_mean_seconds\":{:.6},\"repeats\":{},\"ncv\":{}}}",
        matrix.n_rows,
        matrix.n_cols,
        matrix.data.len(),
        load_secs,
        best,
        sum / repeats as f64,
        repeats,
        ncv.map(|x| x.to_string()).unwrap_or_else(|| "null".to_string())
    );
}
