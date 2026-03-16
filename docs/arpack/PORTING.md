# ARPACK Symmetric Porting Map

This directory tracks the exact ARPACK symmetric path needed by `rupca` to mirror the Scanpy/sklearn sparse PCA route.

Vendored reference sources live in:

- [vendor/arpack-ng/SRC/dsaupd.f](/Users/lpachter/Dropbox/claude/projects/rupca/vendor/arpack-ng/SRC/dsaupd.f)
- [vendor/arpack-ng/SRC/dsaup2.f](/Users/lpachter/Dropbox/claude/projects/rupca/vendor/arpack-ng/SRC/dsaup2.f)
- [vendor/arpack-ng/SRC/dsaitr.f](/Users/lpachter/Dropbox/claude/projects/rupca/vendor/arpack-ng/SRC/dsaitr.f)
- [vendor/arpack-ng/SRC/dsapps.f](/Users/lpachter/Dropbox/claude/projects/rupca/vendor/arpack-ng/SRC/dsapps.f)
- [vendor/arpack-ng/SRC/dsconv.f](/Users/lpachter/Dropbox/claude/projects/rupca/vendor/arpack-ng/SRC/dsconv.f)
- [vendor/arpack-ng/SRC/dseigt.f](/Users/lpachter/Dropbox/claude/projects/rupca/vendor/arpack-ng/SRC/dseigt.f)
- [vendor/arpack-ng/SRC/dsgets.f](/Users/lpachter/Dropbox/claude/projects/rupca/vendor/arpack-ng/SRC/dsgets.f)
- [vendor/arpack-ng/SRC/dseupd.f](/Users/lpachter/Dropbox/claude/projects/rupca/vendor/arpack-ng/SRC/dseupd.f)
- [vendor/arpack-ng/SRC/dsesrt.f](/Users/lpachter/Dropbox/claude/projects/rupca/vendor/arpack-ng/SRC/dsesrt.f)

## Required call graph

For the Scanpy sparse PCA path, the exact symmetric chain is:

1. `dsaupd`
2. `dsaup2`
3. `dsaitr`
4. `dsapps`
5. `dsconv`
6. `dseigt`
7. `dsgets`
8. `dseupd`
9. `dsesrt`

Ancillary helpers that also matter:

- `dsortr`
- `dgetv0`
- selected BLAS/LAPACK helpers:
  - `ddot`
  - `dnrm2`
  - `dcopy`
  - `dscal`
  - `daxpy`
  - `dgemv`
  - `dsteqr`
  - `dgeqr2`
  - `dorm2r`
  - `dlacpy`
  - `dlascl`

## Current status

- The vendored Fortran sources are imported for reference.
- `rupca` currently uses a Rust-native symmetric Krylov/Ritz eigensolver in [src/lib.rs](/Users/lpachter/Dropbox/claude/projects/rupca/src/lib.rs).
- Small direct parity tests against Scanpy/sklearn currently pass, but this is not yet a literal ARPACK routine-by-routine port.

## Translation plan

The intended Rust module split is:

- `src/arpack_symm/state.rs`
  - state carried across the reverse-communication loop
- `src/arpack_symm/dsaupd.rs`
- `src/arpack_symm/dsaup2.rs`
- `src/arpack_symm/dsaitr.rs`
- `src/arpack_symm/dsapps.rs`
- `src/arpack_symm/dsconv.rs`
- `src/arpack_symm/dseigt.rs`
- `src/arpack_symm/dsgets.rs`
- `src/arpack_symm/dseupd.rs`
- `src/arpack_symm/dsesrt.rs`

The first milestone is to replace the current `eigsh(...)` implementation in [src/lib.rs](/Users/lpachter/Dropbox/claude/projects/rupca/src/lib.rs) with a Rust translation of `dsaupd` + `dseupd`, still calling local helpers for the lower-level routine bodies.
