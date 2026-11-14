import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return globally balanced edge-priority indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    eps = np.finfo(float).eps
    off_diagonal = ~np.eye(n, dtype=bool)

    raw = np.where(np.isfinite(distances), distances, np.nan)
    raw = np.maximum(raw, 0.0)
    raw = 0.5 * (raw + raw.T)

    observed = raw[off_diagonal & np.isfinite(raw)]
    positive = observed[observed > 0.0]

    if positive.size:
        fallback = float(np.max(positive))
        global_scale = float(np.median(positive))
    elif observed.size:
        fallback = float(np.max(observed))
        global_scale = max(fallback, 1.0)
    else:
        fallback = 1.0
        global_scale = 1.0

    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0
    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = max(fallback, 1.0)

    d = np.where(np.isfinite(raw), raw, fallback)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)

    def balanced_affinity(temperature: float) -> np.ndarray:
        affinity = np.exp(-np.clip(d / temperature, 0.0, 50.0))
        np.fill_diagonal(affinity, 0.0)

        scaling = np.ones(n, dtype=float)
        for _ in range(120):
            updated = 1.0 / np.maximum(affinity @ scaling, eps)
            updated = np.sqrt(np.maximum(updated * scaling, eps))
            if np.max(np.abs(updated - scaling)) < 1e-10:
                scaling = updated
                break
            scaling = updated

        balanced = scaling[:, None] * affinity * scaling[None, :]
        balanced = 0.5 * (balanced + balanced.T)
        np.fill_diagonal(balanced, 0.0)

        maximum = float(np.max(balanced))
        if maximum > 0.0 and np.isfinite(maximum):
            balanced /= maximum
        return np.nan_to_num(balanced, nan=0.0, posinf=0.0, neginf=0.0)

    broad = balanced_affinity(1.35 * global_scale)
    focused = balanced_affinity(0.55 * global_scale)

    broad_mass = broad / np.maximum(
        np.sum(broad, axis=1, keepdims=True),
        eps,
    )
    focused_mass = focused / np.maximum(
        np.sum(focused, axis=1, keepdims=True),
        eps,
    )

    two_hop = np.sqrt(
        np.maximum(broad_mass @ broad_mass, 0.0)
        * np.maximum(focused_mass @ focused_mass, 0.0)
    )
    two_hop = 0.5 * (two_hop + two_hop.T)
    np.fill_diagonal(two_hop, 0.0)

    maximum = float(np.max(two_hop))
    if maximum > 0.0 and np.isfinite(maximum):
        two_hop /= maximum

    score = np.sqrt(broad * focused)
    score *= 0.78 + 0.22 * two_hop
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(float, copy=False)
