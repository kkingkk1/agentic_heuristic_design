import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric graph-diffusion heuristic matrix for TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    off_diagonal = ~np.eye(n, dtype=bool)
    finite = np.isfinite(distances)
    nonnegative = distances >= 0.0
    valid = finite & nonnegative & off_diagonal
    symmetric_valid = valid & valid.T

    positive = distances[valid & (distances > 0.0)]
    if positive.size == 0:
        return np.zeros((n, n), dtype=float)

    scale = float(np.median(positive))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = float(np.mean(positive))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    finite_values = distances[valid]
    fallback = float(np.percentile(finite_values, 90.0))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = float(np.max(positive))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = scale

    work = np.where(valid, distances, fallback)
    work = np.maximum(work, 0.0)
    work = 0.5 * (work + work.T)
    np.fill_diagonal(work, np.inf)

    eps = np.finfo(float).eps
    finite_work = work[np.isfinite(work)]
    if finite_work.size == 0:
        return np.zeros((n, n), dtype=float)

    positive_work = finite_work[finite_work > 0.0]
    distance_floor = (
        max(float(np.median(positive_work)) * 1e-12, eps)
        if positive_work.size
        else eps
    )
    safe_work = np.maximum(work, distance_floor)

    row_median = np.median(safe_work, axis=1)
    row_median = np.maximum(row_median, distance_floor)

    centered = safe_work - row_median[:, None]
    row_mad = np.median(np.abs(centered), axis=1)
    row_mad = np.maximum(row_mad, distance_floor)

    temperature = row_median + 1.5 * row_mad
    temperature = np.maximum(temperature, distance_floor)

    relative = safe_work / np.sqrt(
        temperature[:, None] * temperature[None, :]
    )

    direct = np.exp(-np.clip(relative, 0.0, 50.0))
    direct[~off_diagonal] = 0.0

    row_sum = np.sum(direct, axis=1, keepdims=True)
    transition = direct / np.maximum(row_sum, eps)

    reverse_consistency = np.sqrt(
        np.maximum(transition * transition.T, 0.0)
    )

    two_step = transition @ transition
    two_step_consistency = np.sqrt(
        np.maximum(two_step * two_step.T, 0.0)
    )

    outgoing_energy = np.sqrt(
        np.sum(transition * transition, axis=1)
    )
    outgoing_energy = np.maximum(outgoing_energy, eps)

    endpoint_balance = np.sqrt(
        outgoing_energy[:, None] * outgoing_energy[None, :]
    )
    endpoint_balance = endpoint_balance / np.maximum(
        np.max(endpoint_balance),
        eps,
    )

    local_density = np.sum(direct, axis=1)
    local_density = np.maximum(local_density, eps)
    density_balance = np.exp(
        -0.20
        * np.abs(
            np.log(
                np.maximum(
                    local_density[:, None] / local_density[None, :],
                    eps,
                )
            )
        )
    )

    direct_factor = np.sqrt(
        np.maximum(
            direct * direct.T,
            0.0,
        )
    )

    closure_factor = 0.82 + 0.18 * np.clip(
        two_step_consistency,
        0.0,
        1.0,
    )

    reciprocity_factor = 0.70 + 0.30 * np.clip(
        reverse_consistency,
        0.0,
        1.0,
    )

    sparsity_factor = 0.88 + 0.12 * np.clip(
        endpoint_balance,
        0.0,
        1.0,
    )

    score = (
        direct_factor
        * reciprocity_factor
        * closure_factor
        * sparsity_factor
        * density_balance
    )

    score = 0.5 * (score + score.T)
    score[~symmetric_valid] = 0.0
    np.fill_diagonal(score, 0.0)

    score = np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
