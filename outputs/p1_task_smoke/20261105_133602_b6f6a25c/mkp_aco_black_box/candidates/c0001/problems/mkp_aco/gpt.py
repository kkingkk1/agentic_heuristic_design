import numpy as np


def heuristics_v2(item_attr1: np.ndarray, item_attr2: np.ndarray) -> np.ndarray:
    """Return a prior-free structural heuristic score for each item."""
    attr1 = np.asarray(item_attr1, dtype=float).reshape(-1)
    attrs2 = np.asarray(item_attr2, dtype=float)

    n = attr1.shape[0]
    if attrs2.ndim == 1:
        attrs2 = attrs2.reshape(n, -1)
    if attrs2.ndim != 2 or attrs2.shape[0] != n:
        raise ValueError("item_attr2 must have shape (n, m)")
    if n == 0:
        return np.empty(0, dtype=float)

    def robust_standardize(x: np.ndarray) -> np.ndarray:
        if x.shape[1] == 0:
            return np.empty_like(x)

        finite = np.isfinite(x)
        med = np.nanmedian(np.where(finite, x, np.nan), axis=0)
        med = np.where(np.isfinite(med), med, 0.0)

        abs_dev = np.abs(x - med)
        mad = np.nanmedian(np.where(np.isfinite(abs_dev), abs_dev, np.nan), axis=0)
        mad = np.where(np.isfinite(mad), mad, 0.0)

        scale = 1.4826 * mad
        fallback = np.nanstd(np.where(finite, x, np.nan), axis=0)
        fallback = np.where(np.isfinite(fallback), fallback, 0.0)
        scale = np.where(scale > 1e-12, scale, fallback)
        scale = np.where(scale > 1e-12, scale, 1.0)

        z = (np.where(finite, x, med) - med) / scale
        return np.clip(z, -8.0, 8.0)

    z1 = robust_standardize(attr1.reshape(-1, 1)).ravel()
    z2 = robust_standardize(attrs2)

    scalar_salience = np.abs(z1)

    if z2.shape[1] > 0:
        feature_salience = np.mean(np.abs(z2), axis=1)
        consistency = np.mean(
            np.tanh(np.abs(z2)) * (np.sign(z2) == np.sign(z2).mean(axis=1, keepdims=True)),
            axis=1,
        )
        consistency = np.maximum(consistency, 0.0)
        representation = np.column_stack((z1, z2))
    else:
        feature_salience = np.zeros(n, dtype=float)
        consistency = np.zeros(n, dtype=float)
        representation = z1.reshape(-1, 1)

    scale = np.sqrt(np.mean(representation**2, axis=0))
    scale = np.where(scale > 1e-12, scale, 1.0)
    representation = representation / scale

    if n == 1:
        density = np.ones(1, dtype=float)
    else:
        squared_norm = np.sum(representation**2, axis=1, keepdims=True)
        distances = squared_norm + squared_norm.T - 2.0 * representation @ representation.T
        distances = np.maximum(distances, 0.0)
        np.fill_diagonal(distances, np.inf)

        k = min(max(1, int(np.sqrt(n))), n - 1)
        nearest = np.partition(distances, kth=k - 1, axis=1)[:, :k]
        local_scale = np.sqrt(np.mean(nearest, axis=1))
        positive = local_scale[np.isfinite(local_scale) & (local_scale > 1e-12)]
        reference = np.median(positive) if positive.size else 1.0
        density = np.exp(-local_scale / max(reference, 1e-12))

    def normalize(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        lo = np.min(x)
        hi = np.max(x)
        if hi - lo <= 1e-12:
            return np.full(x.shape, 0.5, dtype=float)
        return (x - lo) / (hi - lo)

    score = (
        0.35 * normalize(scalar_salience)
        + 0.35 * normalize(feature_salience)
        + 0.20 * normalize(consistency)
        + 0.10 * normalize(density)
    )

    return np.nan_to_num(score, nan=0.0, posinf=1.0, neginf=0.0).astype(float)
