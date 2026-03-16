import json
import sys
import time

from scipy.io import mmread
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA


def main():
    matrix_path = sys.argv[1] if len(sys.argv) > 1 else "/Users/lpachter/Dropbox/claude/projects/rutest/pbmc10k/matrix.mtx"
    n_components = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    repeats = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    start = time.perf_counter()
    x = csr_matrix(mmread(matrix_path))
    load_seconds = time.perf_counter() - start

    best = float("inf")
    total = 0.0
    for _ in range(repeats):
        pca = PCA(n_components=n_components, svd_solver="arpack", random_state=0)
        start = time.perf_counter()
        transformed = pca.fit_transform(x)
        secs = time.perf_counter() - start
        best = min(best, secs)
        total += secs
        transformed.sum()

    print(
        json.dumps(
            {
                "rows": int(x.shape[0]),
                "cols": int(x.shape[1]),
                "nnz": int(x.nnz),
                "load_seconds": load_seconds,
                "pca_best_seconds": best,
                "pca_mean_seconds": total / repeats,
                "repeats": repeats,
            }
        )
    )


if __name__ == "__main__":
    main()
