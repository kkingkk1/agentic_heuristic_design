import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distance_matrix.shape[0]
    scores = np.zeros_like(distance_matrix, dtype=float)

    if n <= 1:
        return scores

    finite = np.isfinite(distance_matrix)
    valid_off_diagonal = finite.copy()
    np.fill_diagonal(valid_off_diagonal, False)

    if not np.any(valid_off_diagonal):
        return scores

    d = np.maximum(distance_matrix, 0.0)
    finite_values = d[valid_off_diagonal]
    fallback_scale = float(np.median(finite_values))
    if not np.isfinite(fallback_scale) or fallback_scale <= 0.0:
        positive_values = finite_values[finite_values > 0.0]
        fallback_scale = (
            float(np.median(positive_values))
            if positive_values.size
            else 1.0
        )

    large = max(float(np.max(finite_values)), fallback_scale, 1.0)
    d = np.where(valid_off_diagonal, d, large * 10.0 + fallback_scale)

    positive_floor = np.finfo(float).eps * max(1.0, fallback_scale)
    d = np.maximum(d, 0.0)

    k = min(3, n - 1)
    sorted_rows = np.sort(
        np.where(valid_off_diagonal, d, np.inf),
        axis=1,
    )
    local_samples = sorted_rows[:, :k]

    row_scale = np.full(n, fallback_scale, dtype=float)
    for i in range(n):
        values = local_samples[i]
        values = values[np.isfinite(values)]
        if values.size:
            positive = values[values > 0.0]
            if positive.size:
                row_scale[i] = float(np.median(positive))
            else:
                row_scale[i] = float(np.median(values))

    row_scale = np.maximum(row_scale, positive_floor)
    column_scale = row_scale.copy()

    forward_affinity = 1.0 / (1.0 + d / row_scale[:, None])
    reverse_affinity = 1.0 / (1.0 + d / column_scale[None, :])
    affinity = forward_affinity * reverse_affinity

    np.fill_diagonal(affinity, 0.0)
    affinity[~valid_off_diagonal] = 0.0

    profiles = affinity.copy()
    profile_norms = np.linalg.norm(profiles, axis=1, keepdims=True)
    profile_norms = np.maximum(profile_norms, positive_floor)
    normalized_profiles = profiles / profile_norms

    compatibility = normalized_profiles @ normalized_profiles.T
    compatibility = np.clip(compatibility, 0.0, 1.0)
    np.fill_diagonal(compatibility, 0.0)

    scores = affinity * (0.5 + 0.5 * compatibility)
    np.fill_diagonal(scores, 0.0)
    scores[~valid_off_diagonal] = 0.0

    if np.allclose(
        distance_matrix,
        distance_matrix.T,
        rtol=1e-10,
        atol=1e-12,
        equal_nan=True,
    ):
        scores = 0.99 * scores + 0.01 * scores.T

    maximum = float(np.max(scores))
    if maximum > 0.0 and np.isfinite(maximum):
        scores /= maximum

    return scores
