import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Construct symmetric edge priors for stochastic TSP solution sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    off_diagonal = ~np.eye(n, dtype=bool)
    finite_off_diagonal = np.isfinite(d) & off_diagonal
    finite_values = d[finite_off_diagonal]

    if finite_values.size == 0:
        return np.zeros((n, n), dtype=float)

    finite_values = np.maximum(finite_values, 0.0)
    positive_values = finite_values[finite_values > 0.0]

    global_scale = (
        float(np.median(positive_values))
        if positive_values.size
        else max(float(np.median(finite_values)), 1.0)
    )
    global_scale = max(global_scale, np.finfo(float).eps)

    replacement = float(np.max(finite_values) + 10.0 * global_scale)

    clean = np.where(np.isfinite(d), np.maximum(d, 0.0), replacement)
    clean = 0.5 * (clean + clean.T)
    np.fill_diagonal(clean, replacement)

    eps = np.finfo(float).eps * max(global_scale, 1.0)

    row_sorted = np.sort(clean, axis=1)
    nearest = row_sorted[:, 0]
    local_count = max(1, min(n - 1, int(np.sqrt(n)) + 1))
    local_scale = np.median(row_sorted[:, :local_count], axis=1)

    row_median = np.median(clean, axis=1)
    row_spread = np.median(
        np.abs(clean - row_median[:, None]),
        axis=1,
    )

    temperature = np.maximum(
        row_spread + 0.05 * np.maximum(local_scale, eps),
        eps,
    )

    preference = np.exp(
        -np.minimum(
            np.maximum(clean - nearest[:, None], 0.0)
            / temperature[:, None],
            60.0,
        )
    )
    preference[~off_diagonal] = 0.0

    row_mass = preference.sum(axis=1, keepdims=True)
    row_mass = np.maximum(row_mass, eps)
    preference /= row_mass

    mutual_preference = np.sqrt(
        np.maximum(preference, 0.0)
        * np.maximum(preference.T, 0.0)
    )

    regret = np.maximum(clean - nearest[:, None], 0.0)
    regret /= np.maximum(row_median[:, None], global_scale * 0.05)
    scarcity = np.exp(-np.minimum(regret, 60.0))
    scarcity = np.sqrt(scarcity * scarcity.T)

    global_attraction = np.exp(
        -np.minimum(clean / global_scale, 60.0)
    )

    density = 1.0 / np.maximum(local_scale, eps)
    density /= max(float(np.max(density)), eps)
    degree_balance = np.sqrt(
        np.outer(density, density)
    )
    degree_balance = 0.85 + 0.15 * degree_balance

    score = (
        0.40 * global_attraction
        + 0.35 * mutual_preference
        + 0.25 * scarcity
    )
    score *= degree_balance
    score = 0.5 * (score + score.T)

    score[~finite_off_diagonal] = 0.0
    np.fill_diagonal(score, 0.0)
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
