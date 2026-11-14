import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return diffusion- and spectrum-aware edge priors for stochastic TSP."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    finite_nonnegative = np.isfinite(raw) & (raw >= 0.0)
    valid = finite_nonnegative & finite_nonnegative.T & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    distances = np.full((n, n), np.inf, dtype=float)
    distances[valid] = 0.5 * (
        raw[valid] + raw.T[valid]
    )
    np.fill_diagonal(distances, np.inf)

    values = distances[valid]
    positive = values[values > 0.0]

    if positive.size:
        center = float(np.median(positive))
        mad = float(np.median(np.abs(positive - center)))
    else:
        center = 1.0
        mad = 0.0

    if not np.isfinite(center) or center <= 0.0:
        center = 1.0

    scale = max(
        mad,
        0.20 * center,
        np.finfo(float).eps,
    )

    lower = float(np.quantile(values, 0.10))
    upper = float(np.quantile(values, 0.90))
    spread = max(upper - lower, scale)

    replacement = upper + 10.0 * spread
    clean = np.where(valid, distances, replacement)
    np.fill_diagonal(clean, replacement)

    eps = np.finfo(float).eps

    ranks = np.empty_like(clean)
    order = np.argsort(clean, axis=1, kind="stable")
    ranks[
        np.arange(n)[:, None],
        order,
    ] = np.arange(n, dtype=float)

    rank_affinity = np.exp(
        -ranks / max(1.0, 0.75 * np.sqrt(float(n)))
    )
    rank_affinity[~valid] = 0.0

    distance_affinity = np.zeros((n, n), dtype=float)
    distance_affinity[valid] = np.exp(
        -np.clip(
            (clean[valid] - lower) / max(scale, eps),
            0.0,
            50.0,
        )
    )

    direct = (
        np.maximum(rank_affinity, eps) ** 0.42
        * np.maximum(distance_affinity, eps) ** 0.58
    )
    direct[~valid] = 0.0

    row_mass = direct.sum(axis=1)
    transition = direct / np.maximum(row_mass[:, None], eps)

    incoming = transition.sum(axis=0)
    stationary = incoming / max(float(np.sum(incoming)), eps)
    stationary = np.maximum(stationary, eps)
    stationary /= np.sum(stationary)

    occupancy = np.sqrt(
        stationary[:, None] * stationary[None, :]
    )

    profile_overlap = transition @ transition.T
    profile_norm = np.sqrt(
        np.maximum(
            np.sum(transition * transition, axis=1)[:, None]
            * np.sum(transition * transition, axis=1)[None, :],
            eps,
        )
    )
    profile_similarity = profile_overlap / profile_norm
    profile_similarity[~valid] = 0.0

    two_step = transition @ transition
    three_step = two_step @ transition
    closure = np.sqrt(
        np.maximum(
            two_step * two_step.T,
            0.0,
        )
    )
    closure = closure + 0.5 * np.sqrt(
        np.maximum(
            three_step * three_step.T,
            0.0,
        )
    )
    closure[~valid] = 0.0

    degree = direct.sum(axis=1)
    inv_sqrt_degree = 1.0 / np.sqrt(
        np.maximum(degree, eps)
    )
    normalized_adjacency = (
        inv_sqrt_degree[:, None]
        * direct
        * inv_sqrt_degree[None, :]
    )
    normalized_adjacency = 0.5 * (
        normalized_adjacency
        + normalized_adjacency.T
    )

    spectral_similarity = np.zeros((n, n), dtype=float)

    try:
        eigenvalues, eigenvectors = np.linalg.eigh(
            np.eye(n) - normalized_adjacency
        )
        count = min(5, max(1, n - 1))
        selected = eigenvectors[:, :count]

        pairwise = selected[:, None, :] - selected[None, :, :]
        squared_distance = np.sum(pairwise * pairwise, axis=2)

        spectral_scale = float(
            np.median(squared_distance[valid])
        )
        if not np.isfinite(spectral_scale) or spectral_scale <= eps:
            spectral_scale = 1.0

        spectral_similarity[valid] = np.exp(
            -squared_distance[valid]
            / spectral_scale
        )
    except np.linalg.LinAlgError:
        spectral_similarity[valid] = 1.0

    spectral_similarity[~valid] = 0.0

    direct_max = max(float(np.max(direct)), eps)
    closure_max = max(float(np.max(closure)), eps)
    profile_max = max(float(np.max(profile_similarity)), eps)
    spectral_max = max(float(np.max(spectral_similarity)), eps)

    direct_signal = direct / direct_max
    closure_signal = closure / closure_max
    profile_signal = profile_similarity / profile_max
    spectral_signal = spectral_similarity / spectral_max

    score = (
        np.maximum(direct_signal, eps) ** 0.48
        * np.maximum(closure_signal + 0.08, eps) ** 0.20
        * np.maximum(profile_signal + 0.12, eps) ** 0.17
        * np.maximum(spectral_signal + 0.10, eps) ** 0.15
        * np.maximum(occupancy, eps) ** 0.08
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if not np.isfinite(maximum) or maximum <= 0.0:
        result = valid.astype(float)
        np.fill_diagonal(result, 0.0)
        return result

    result = score / maximum
    result[~valid] = 0.0
    np.fill_diagonal(result, 0.0)

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(float, copy=False)
