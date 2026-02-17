from __future__ import annotations
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IN_PATH = PROJECT_ROOT / "data" / "out" / "drivers_encoded.csv"
OUT_NPY = PROJECT_ROOT / "data" / "out" / "drivers_gower.npy"
OUT_META = PROJECT_ROOT / "data" / "out" / "drivers_gower_meta.json"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def compute_gower_numeric_with_missing(
    X: np.ndarray,
    feature_ranges: np.ndarray,
    block_size: int = 256,
) -> np.ndarray:
    """
    Numeric-only Gower distance WITH missing values (NaN) supported.

    For each pair (i, j), compute:
        d(i,j) = mean_k( |xik - xjk| / range_k ) over features k where:
                 - range_k > 0
                 - both xik and xjk are observed (not NaN)

    If a pair has zero comparable features (should be rare), distance is set to 1.0.

    This is the correct Gower-style handling of missingness: do NOT impute, instead
    compute distances using available comparable dimensions.
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D; got shape {X.shape}")

    n, p = X.shape
    if feature_ranges.shape != (p,):
        raise ValueError(f"feature_ranges must have shape ({p},); got {feature_ranges.shape}")

    # Only non-constant, well-defined ranges contribute
    valid_feat = np.isfinite(feature_ranges) & (feature_ranges > 0)
    if valid_feat.sum() == 0:
        raise ValueError("No valid features (all ranges are 0 or NaN). Gower undefined.")

    Xv = X[:, valid_feat].astype(np.float64, copy=False)
    rv = feature_ranges[valid_feat].astype(np.float64, copy=False)

    n2, p2 = Xv.shape
    assert n2 == n

    D = np.zeros((n, n), dtype=np.float64)

    # Precompute finite mask for Xv to avoid repeated np.isfinite calls
    finite_mask = np.isfinite(Xv)  # (n, p2)

    for i0 in range(0, n, block_size):
        i1 = min(n, i0 + block_size)
        A = Xv[i0:i1, :]                 # (b, p2)
        A_fin = finite_mask[i0:i1, :]    # (b, p2)

        # Compute normalized absolute differences for all pairs block vs all rows
        # diff: (b, n, p2)
        diff = np.abs(A[:, None, :] - Xv[None, :, :]) / rv[None, None, :]

        # comparable mask: both finite
        comp = A_fin[:, None, :] & finite_mask[None, :, :]  # (b, n, p2)

        # zero-out non-comparable contributions
        diff = np.where(comp, diff, 0.0)

        # count comparable features per pair
        denom = comp.sum(axis=2).astype(np.float64)  # (b, n)

        # sum of contributions
        numer = diff.sum(axis=2)  # (b, n)

        # average over comparable features
        block = np.empty((i1 - i0, n), dtype=np.float64)

        # where denom > 0, compute mean; else set to 1.0 (max dissimilarity)
        good = denom > 0
        block[good] = numer[good] / denom[good]
        block[~good] = 1.0

        D[i0:i1, :] = block

    # Force symmetry and diagonal zero (important for scipy squareform/linkage)
    D = (D + D.T) / 2.0
    np.fill_diagonal(D, 0.0)

    return D


def main(force: bool = False, block_size: int = 256) -> None:
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Missing input encoded drivers: {IN_PATH}")

    in_hash = sha256_file(IN_PATH)

    if OUT_NPY.exists() and OUT_META.exists() and not force:
        try:
            meta = json.loads(OUT_META.read_text(encoding="utf-8"))
            if meta.get("input_sha256") == in_hash:
                print("Gower distance already computed for this exact drivers_encoded.csv.")
                print("Using cached:", OUT_NPY)
                return
            else:
                print("Found cached Gower, but input hash changed -> recomputing.")
        except Exception:
            print("Found cached files but metadata unreadable -> recomputing.")

    df = pd.read_csv(IN_PATH)
    if "respondent_id" in df.columns:
        df = df.drop(columns=["respondent_id"])

    # Require numeric dtypes (NaNs allowed)
    X = df.apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)

    # Compute ranges ignoring NaNs (required for ordinal columns with NaN by design)
    col_min = np.nanmin(X, axis=0)
    col_max = np.nanmax(X, axis=0)
    ranges = (col_max - col_min).astype(np.float64)

    n, p = X.shape
    print(f"Computing Gower (numeric-only, missing-aware) for n={n}, p={p}")
    print(f"Output distance matrix shape: ({n}, {n})")
    print(f"Block size: {block_size}")

    D64 = compute_gower_numeric_with_missing(X=X, feature_ranges=ranges, block_size=block_size)

    # Diagnostics
    if np.isnan(D64).any():
        n_nan = int(np.isnan(D64).sum())
        raise ValueError(f"Distance matrix contains NaNs after missing-aware Gower: {n_nan} NaNs")

    max_asym = float(np.max(np.abs(D64 - D64.T)))
    diag_max = float(np.max(np.abs(np.diag(D64))))
    dmin = float(np.min(D64))
    dmax = float(np.max(D64))

    print(f"Max |D-D.T| after symmetrize: {max_asym:.3e}")
    print(f"Max |diag(D)|: {diag_max:.3e}")
    print(f"Distance range: min={dmin:.6f}, max={dmax:.6f}")

    # Store compactly
    D = D64.astype(np.float32)

    OUT_NPY.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT_NPY, D)

    meta = {
        "input_path": str(IN_PATH),
        "input_sha256": in_hash,
        "n_rows": int(n),
        "n_features": int(p),
        "dtype_saved": "float32",
        "dtype_computed": "float64",
        "block_size": int(block_size),
        "missingness_handling": (
            "Pairwise mean over comparable (non-NaN) features only; "
            "features with zero range excluded; if a pair has 0 comparable features, distance=1.0."
        ),
        "note": (
            "Gower-style distance computed as mean(|xi-xj|/range) over non-constant features, "
            "ignoring NaNs pairwise (no imputation). Encoded 0/1 flags + ordinals treated as numeric."
        ),
    }
    OUT_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Saved:", OUT_NPY)
    print("Saved metadata:", OUT_META)


if __name__ == "__main__":
    # Set force=True once if you want to overwrite an older cached distance matrix.
    main(force=False, block_size=256)
