import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-priority indicators for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite_offdiag = np.isfinite(raw) & ~np.eye(n, dtype=bool)
    positive = raw[finite_offdiag & (raw > 0)]

    if positive.size == 0:
        return np.zeros((n, n), dtype=float)

    base_scale = float(np.median(positive))
    base_scale = max(base_scale, np.finfo(float).eps)

    d = np.where(np.isfinite(raw), np.maximum(raw, 0.0), np.inf)
    np.fill_diagonal(d, np.inf)

    fallback = float(np.max(positive))
    finite_d = np.where(np.isfinite(d), d, fallback * 4.0)

    positive_rows = np.where(
        np.isfinite(d) & (d > 0),
        d,
        np.nan,
    )
    local_scale = np.nanmedian(positive_rows, axis=1)
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0),
        local_scale,
        base_scale,
    )

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    pair_scale = np.maximum(pair_scale, np.finfo(float).eps)

    direct = finite_d
    valid = np.isfinite(d) & ~np.eye(n, dtype=bool)

    # For each edge, compare its direct length with all two-hop alternatives.
    detour = direct[:, :, None] - 0.0
    two_hop = finite_d[:, None, :] + finite_d[None, :, :]
    two_hop[:, :, np.arange(n)] = np.inf
    two_hop[np.arange(n), np.arange(n), :] = np.inf

    gain = (two_hop - direct[:, :, None]) / (
        two_hop + direct[:, :, None] + base_scale
    )
    gain = np.clip(gain, -1.0, 1.0)

    alternatives = np.isfinite(two_hop)
    gain = np.where(alternatives, gain, np.nan)

    if n > 2:
        closure = np.nanpercentile(gain, 25.0, axis=2)
    else:
        closure = np.zeros((n, n), dtype=float)

    closure = np.nan_to_num(closure, nan=0.0, posinf=0.0, neginf=0.0)
    closure = np.maximum(closure, 0.0)

    # Prefer edges whose endpoints have compatible local distance scales.
    scale_match = np.exp(
        -np.abs(np.log(
            (local_scale[:, None] + base_scale)
            / (local_scale[None, :] + base_scale)
        ))
    )

    # Convert triangle advantage into a positive structural support.
    structural = 0.5 + 0.5 * closure
    structural *= scale_match

    # Softly penalize edges incident to nodes with excessive total support.
    support = np.sum(np.where(valid, structural, 0.0), axis=1)
    support = np.maximum(support, np.finfo(float).eps)
    balance = 1.0 / np.sqrt(support[:, None] * support[None, :])

    score = structural * balance
    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
