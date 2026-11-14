import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return diffusion-based edge priorities for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps

    finite = np.isfinite(distances)
    positive = distances[finite & (distances > 0.0)]

    if positive.size:
        fallback = float(np.percentile(positive, 90.0))
        scale = float(np.median(positive))
    else:
        fallback = 1.0
        scale = 1.0

    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = fallback

    d = np.where(finite, distances, fallback)
    d = np.nan_to_num(
        d,
        nan=fallback,
        posinf=fallback,
        neginf=0.0,
    )
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)

    off_diagonal = ~np.eye(n, dtype=bool)

    observed = d[off_diagonal]
    if observed.size == 0:
        return np.zeros((n, n), dtype=float)

    positive_observed = observed[observed > 0.0]
    if positive_observed.size:
        low = float(np.percentile(positive_observed, 20.0))
        high = float(np.percentile(positive_observed, 80.0))
    else:
        low = 0.0
        high = fallback

    spread = max(high - low, eps)
    relative = np.maximum(d - low, 0.0) / spread

    # A bounded distance kernel avoids domination by a few extreme entries.
    kernel = np.exp(-np.clip(relative, 0.0, 30.0))
    kernel[~off_diagonal] = 0.0

    row_mass = np.sum(kernel, axis=1)
    row_mass = np.maximum(row_mass, eps)

    transition = kernel / row_mass[:, None]

    # Symmetric flow measures agreement between the two directed transitions.
    reverse_transition = transition.T
    flow = np.sqrt(
        np.maximum(transition, 0.0)
        * np.maximum(reverse_transition, 0.0)
    )

    # Two-step support rewards edges embedded in coherent short paths.
    two_step = transition @ transition
    two_step = np.maximum(two_step, 0.0)

    two_step_scale = np.sqrt(
        np.maximum(two_step, 0.0)
        * np.maximum(two_step.T, 0.0)
    )

    # Estimate a stationary visitation profile by a few diffusion steps.
    stationary = np.full(n, 1.0 / n, dtype=float)
    for _ in range(4):
        stationary = stationary @ transition
        total = float(np.sum(stationary))
        if not np.isfinite(total) or total <= eps:
            stationary.fill(1.0 / n)
            break
        stationary /= total

    visitation_balance = np.sqrt(
        np.maximum(stationary[:, None], eps)
        * np.maximum(stationary[None, :], eps)
    )

    max_balance = float(np.max(visitation_balance))
    if max_balance > eps and np.isfinite(max_balance):
        visitation_balance /= max_balance

    # Penalize edges whose endpoints have very different transition entropy.
    entropy = -np.sum(
        np.where(
            transition > 0.0,
            transition * np.log(np.maximum(transition, eps)),
            0.0,
        ),
        axis=1,
    )
    entropy_scale = max(float(np.mean(entropy)), eps)
    entropy_balance = np.exp(
        -np.abs(entropy[:, None] - entropy[None, :])
        / entropy_scale
    )

    direct = np.sqrt(
        np.maximum(transition, 0.0)
        * np.maximum(reverse_transition, 0.0)
    )

    closure_scale = np.maximum(
        np.mean(two_step_scale[off_diagonal])
        if np.any(off_diagonal)
        else 1.0,
        eps,
    )
    closure = np.tanh(two_step_scale / closure_scale)

    score = (
        direct ** 0.90
        * (0.55 + 0.45 * closure)
        * (0.70 + 0.30 * entropy_balance)
        * (0.65 + 0.35 * visitation_balance)
        * (0.75 + 0.25 * kernel)
    )

    score = 0.5 * (score + score.T)
    score[~off_diagonal] = 0.0

    score = np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    values = score[off_diagonal]
    if values.size:
        low_score = float(np.percentile(values, 5.0))
        high_score = float(np.percentile(values, 95.0))
        score = np.clip(
            (score - low_score) / max(high_score - low_score, eps),
            0.0,
            1.0,
        )

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
