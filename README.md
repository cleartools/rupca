# rupca (rust PCA)

`rupca` is a small Rust crate that mirrors the sparse centered PCA path used by Scanpy for sparse input with `zero_center=True`.

The implemented path is:

1. Compute per-feature means and variances.
2. Represent centering implicitly as `X - 1 * mean^T`.
3. Build the normal operator on the smaller side:
   - `(X - mean)^T (X - mean)` when `n_samples >= n_features`
   - `(X - mean) (X - mean)^T` otherwise
4. Run a Rust-native symmetric Lanczos/Ritz eigensolver on that implicit operator.
5. Form `Av` exactly as SciPy does.
6. Run dense SVD on `Av`.
7. Recover scores, components, singular values, explained variance, and noise variance in the same style as sklearn PCA.

The current public entrypoint is:

- `pca_scanpy_sparse_csr(&CsrMatrix, ScanpyPcaParams) -> Result<ScanpyPcaResult>`

The matrix format is a simple CSR container owned by `rupca`.

## Status

The crate currently:

- compiles cleanly
- passes unit tests on both tall and wide sparse matrices against the corresponding centered dense SVD reference
- vendors the exact ARPACK symmetric reference sources used for the Scanpy/sklearn sparse path in [vendor/arpack-ng/SRC](/Users/lpachter/Dropbox/claude/projects/rupca/vendor/arpack-ng/SRC)

## Notes

- This is intended to match the Scanpy sparse PCA algorithmic path, not the full Scanpy Python object model.
- The eigensolver is now Rust-native rather than calling external ARPACK.
- The imported ARPACK reference sources and porting map are documented in [docs/arpack/PORTING.md](/Users/lpachter/Dropbox/claude/projects/rupca/docs/arpack/PORTING.md).
