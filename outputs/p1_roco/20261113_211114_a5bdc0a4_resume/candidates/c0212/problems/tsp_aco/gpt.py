import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return graph-spectral edge-selection priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    finite_nonnegative = np.isfinite(raw) & (raw >= 0.0)
    valid = finite_nonnegative & finite_nonnegative.T & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps

    symmetric = np.full((n, n), np.inf, dtype=float)
    symmetric[valid] = 0.5 * (raw[valid] + raw.T[valid])
    np.fill_diagonal(symmetric, np.inf)

    edge_values = symmetric[valid]
    positive = edge_values[edge_values > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    # Robust global rank representation. It is insensitive to absolute units.
    ordered = np.sort(edge_values)
    ranks = np.searchsorted(ordered, edge_values, side="right").astype(float)
    percentile = ranks / max(float(edge_values.size), 1.0)

    rank_affinity = np.zeros((n, n), dtype=float)
    rank_affinity[valid] = 1.0 - percentile
    rank_affinity[valid] = 0.08 + 0.92 * rank_affinity[valid]

    # Convert distances into conductances for graph diffusion.
    clipped = np.maximum(symmetric, 0.0)
    robust_temperature = max(
        float(np.median(np.abs(edge_values - np.median(edge_values)))),
        0.18 * scale,
        eps,
    )

    conductance = np.zeros((n, n), dtype=float)
    conductance[valid] = np.exp(
        -np.minimum(clipped[valid] / robust_temperature, 50.0)
    )

    degree = conductance.sum(axis=1)
    active = degree > 0.0

    if not np.any(active):
        result = valid.astype(float)
        np.fill_diagonal(result, 0.0)
        maximum = float(np.max(result))
        return result / maximum if maximum > 0.0 else result

    safe_degree = np.maximum(degree, eps)
    inv_sqrt_degree = 1.0 / np.sqrt(safe_degree)

    normalized_adjacency = (
        inv_sqrt_degree[:, None] * conductance * inv_sqrt_degree[None, :]
    )

    # Low-frequency eigenvectors describe global graph geometry.
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(
            np.eye(n, dtype=float) - normalized_adjacency
        )
        usable = np.isfinite(eigenvalues) & np.all(
            np.isfinite(eigenvectors), axis=0
        )
        usable_indices = np.flatnonzero(usable)

        if usable_indices.size:
            embedding_count = min(
                max(2, int(np.ceil(np.sqrt(n)))),
                usable_indices.size,
            )
            selected = usable_indices[:embedding_count]
            coordinates = eigenvectors[:, selected]
            spectral_distance = np.linalg.norm(
                coordinates[:, None, :] - coordinates[None, :, :],
                axis=2,
            )

            spectral_scale = float(
                np.median(spectral_distance[valid])
            )
            spectral_scale = max(spectral_scale, eps)

            spectral_affinity = np.zeros((n, n), dtype=float)
            spectral_affinity[valid] = np.exp(
                -np.minimum(
                    spectral_distance[valid] / spectral_scale,
                    40.0,
                )
            )
        else:
            spectral_affinity = valid.astype(float)
    except np.linalg.LinAlgError:
        spectral_affinity = valid.astype(float)

    # A short diffusion walk rewards edges joining nodes with compatible
    # multi-step access patterns, without explicitly enumerating paths.
    transition = conductance / safe_degree[:, None]
    diffusion = transition @ transition.T
    diffusion = np.maximum(diffusion, 0.0)

    diffusion_scale = float(np.max(diffusion[valid]))
    if not np.isfinite(diffusion_scale) or diffusion_scale <= 0.0:
        diffusion_scale = 1.0

    diffusion_affinity = np.zeros((n, n), dtype=float)
    diffusion_affinity[valid] = (
        0.15
        + 0.85 * np.sqrt(
            np.clip(diffusion[valid] / diffusion_scale, 0.0, 1.0)
        )
    )

    # Normalize direct distance independently from the graph-derived terms.
    direct_affinity = np.zeros((n, n), dtype=float)
    direct_affinity[valid] = np.exp(
        -np.minimum(clipped[valid] / max(0.65 * scale, eps), 50.0)
    )
    direct_affinity[valid] = 0.12 + 0.88 * direct_affinity[valid]

    degree_balance = np.sqrt(
        np.minimum(
            degree[:, None] / np.maximum(np.max(degree), eps),
            degree[None, :] / np.maximum(np.max(degree), eps),
        )
    )
    degree_balance = 0.35 + 0.65 * degree_balance

    score = (
        np.maximum(rank_affinity, eps) ** 0.30
        * np.maximum(direct_affinity, eps) ** 0.24
        * np.maximum(spectral_affinity, eps) ** 0.28
        * np.maximum(diffusion_affinity, eps) ** 0.12
        * np.maximum(degree_balance, eps) ** 0.06
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    score = np.maximum(score, 0.0)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum
    else:
        score.fill(0.0)

    return np.nan_to_num(
        score.astype(float, copy=False),
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
