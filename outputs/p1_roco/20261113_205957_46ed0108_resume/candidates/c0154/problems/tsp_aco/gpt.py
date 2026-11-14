import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge priors from similarity between global distance profiles."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = raw.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    valid = np.isfinite(raw) & (raw >= 0.0) & ~diagonal
    mutual_valid = valid & valid.T

    if not np.any(mutual_valid):
        return np.zeros((n, n), dtype=float)

    observed = raw[mutual_valid]
    positive = observed[observed > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    largest = float(np.max(observed)) if observed.size else scale
    penalty = largest + 12.0 * scale

    clean = np.where(mutual_valid, 0.5 * (raw + raw.T), penalty)
    np.fill_diagonal(clean, penalty)

    row_median = np.median(
        np.where(mutual_valid, clean, np.nan),
        axis=1,
    )
    row_median = np.nan_to_num(
        row_median,
        nan=scale,
        posinf=scale,
        neginf=scale,
    )

    centered = clean - row_median[:, None]
    profile_scale = np.median(
        np.abs(np.where(mutual_valid, centered, np.nan)),
        axis=1,
    )
    profile_scale = np.nan_to_num(
        profile_scale,
        nan=0.5 * scale,
        posinf=0.5 * scale,
        neginf=0.5 * scale,
    )
    profile_scale = np.maximum(profile_scale, 0.08 * scale)

    z = centered / profile_scale[:, None]
    response = np.exp(-0.5 * np.clip(z, -12.0, 12.0) ** 2)
    response[~mutual_valid] = 0.0
    np.fill_diagonal(response, 0.0)

    norms = np.sqrt(np.sum(response * response, axis=1))
    safe_norms = np.maximum(norms, np.finfo(float).eps)
    similarity = (response @ response.T) / (
        safe_norms[:, None] * safe_norms[None, :]
    )

    profile_mass = response.sum(axis=1)
    mass_scale = max(float(np.median(profile_mass)), np.finfo(float).eps)
    centrality = np.sqrt(
        np.minimum(profile_mass[:, None], profile_mass[None, :])
        / mass_scale
    )
    centrality = np.clip(centrality, 0.35, 1.65)

    local_agreement = np.exp(
        -np.abs(row_median[:, None] - row_median[None, :])
        / max(scale, np.finfo(float).eps)
    )

    score = similarity * centrality * (0.55 + 0.45 * local_agreement)
    score = 0.5 * (score + score.T)
    score[~mutual_valid] = 0.0
    score = np.maximum(score, 0.0)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if not np.isfinite(maximum) or maximum <= 0.0:
        return np.zeros((n, n), dtype=float)

    return (score / maximum).astype(float, copy=False)
