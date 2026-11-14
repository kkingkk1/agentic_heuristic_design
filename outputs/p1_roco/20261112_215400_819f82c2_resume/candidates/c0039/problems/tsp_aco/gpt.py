import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")
    if np.any(~np.isfinite(d)) or np.any(d < 0):
        raise ValueError("distance_matrix must contain finite nonnegative values")

    n = d.shape[0]
    if n <= 1:
        return np.zeros_like(d, dtype=float)

    d = 0.5 * (d + d.T)
    work = d.copy()
    np.fill_diagonal(work, np.inf)

    off_diag = ~np.eye(n, dtype=bool)
    values = work[off_diag]
    if values.size == 0:
        return np.zeros((n, n), dtype=float)

    q25, q50, q75 = np.percentile(values, [25.0, 50.0, 75.0])
    spread = max(float(q75 - q25), float(np.std(values)), np.finfo(float).eps)

    centered = (q50 - work) / spread
    centered = np.clip(centered, -50.0, 50.0)
    preference = 1.0 / (1.0 + np.exp(-centered))
    preference[~np.isfinite(preference)] = 0.0
    np.fill_diagonal(preference, 0.0)

    row_mass = preference.sum(axis=1, keepdims=True)
    transition = np.divide(
        preference,
        row_mass,
        out=np.zeros_like(preference),
        where=row_mass > 0,
    )

    endpoint_agreement = np.sqrt(transition * transition.T)

    i_idx = np.arange(n)[:, None, None]
    k_idx = np.arange(n)[None, :, None]
    j_idx = np.arange(n)[None, None, :]

    two_hop = work[:, :, None] + work[None, :, :]
    invalid_intermediates = (k_idx == i_idx) | (k_idx == j_idx)
    two_hop = np.where(invalid_intermediates, np.inf, two_hop)

    best_detour = np.min(two_hop, axis=1)
    finite_detours = np.isfinite(two_hop)
    detour_count = np.sum(finite_detours, axis=1)

    detour_sum = np.sum(
        np.where(finite_detours, two_hop, 0.0),
        axis=1,
    )
    typical_detour = np.divide(
        detour_sum,
        detour_count,
        out=np.full((n, n), q50 + spread, dtype=float),
        where=detour_count > 0,
    )

    best_margin = (best_detour - work) / spread
    typical_margin = (typical_detour - work) / spread

    best_margin = np.clip(best_margin, -50.0, 50.0)
    typical_margin = np.clip(typical_margin, -50.0, 50.0)

    best_advantage = 1.0 / (1.0 + np.exp(-best_margin))
    typical_advantage = 1.0 / (1.0 + np.exp(-typical_margin))

    detour_quality = 0.65 * best_advantage + 0.35 * typical_advantage
    detour_quality[~np.isfinite(detour_quality)] = 0.0

    score = (
        (0.35 + 0.65 * preference)
        * (0.35 + 0.65 * endpoint_agreement)
        * (0.25 + 0.75 * detour_quality)
    )

    score = 0.5 * (score + score.T)
    score[~np.isfinite(score)] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score.astype(float, copy=False)
