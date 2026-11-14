import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return deterministic diffusion-cycle priors for directed TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    valid = np.isfinite(raw) & (raw >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    values = raw[valid]
    q10, q50, q90 = np.quantile(values, [0.10, 0.50, 0.90])

    spread = float(q90 - q10)
    positive = values[values > 0.0]

    if not np.isfinite(spread) or spread <= 0.0:
        spread = float(np.median(np.abs(values - q50)))

    if not np.isfinite(spread) or spread <= 0.0:
        spread = float(np.mean(positive)) if positive.size else 1.0

    scale = max(spread, np.finfo(float).eps)
    upper = max(float(q90), float(q50), scale)
    replacement = upper + 18.0 * scale

    clean = np.full((n, n), replacement, dtype=float)
    clean[valid] = raw[valid]
    np.fill_diagonal(clean, replacement)

    temperature = max(0.35 * scale, np.finfo(float).eps)

    shifted = clean - np.min(clean[valid])
    kernel = np.exp(-np.minimum(shifted / temperature, 50.0))
    kernel[~valid] = 0.0

    eps = np.finfo(float).eps

    # Row-normalized directed transition kernel.
    row_mass = kernel.sum(axis=1, keepdims=True)
    transition = np.divide(
        kernel,
        np.maximum(row_mass, eps),
        out=np.zeros_like(kernel),
        where=row_mass > 0.0,
    )

    # Iterative matrix scaling creates a direction-preserving global edge flow.
    left_scale = np.ones(n, dtype=float)
    right_scale = np.ones(n, dtype=float)

    for _ in range(48):
        column_mass = kernel.T @ left_scale
        right_scale = np.divide(
            1.0,
            np.maximum(column_mass, eps),
            out=np.zeros_like(column_mass),
            where=column_mass > 0.0,
        )

        row_mass_scaled = kernel @ right_scale
        left_scale = np.divide(
            1.0,
            np.maximum(row_mass_scaled, eps),
            out=np.zeros_like(row_mass_scaled),
            where=row_mass_scaled > 0.0,
        )

    balanced = kernel * left_scale[:, None] * right_scale[None, :]
    balanced[~valid] = 0.0

    balanced_max = float(np.max(balanced))
    if np.isfinite(balanced_max) and balanced_max > 0.0:
        balanced /= balanced_max

    # A directed edge i -> j is useful when j can return to i in two steps.
    two_step = transition @ transition
    cycle_support = two_step.T.copy()
    cycle_support[~valid] = 0.0

    cycle_max = float(np.max(cycle_support))
    if np.isfinite(cycle_max) and cycle_max > 0.0:
        cycle_support /= cycle_max

    # Diffusion reachability estimates whether the destination lies in a
    # broadly accessible region of the directed graph.
    diffusion = transition.copy()
    reachability = transition.copy()

    for _ in range(5):
        diffusion = diffusion @ transition
        reachability += diffusion

    destination_mass = np.sum(reachability, axis=0)
    source_mass = np.sum(reachability, axis=1)

    destination_mass /= max(float(np.max(destination_mass)), eps)
    source_mass /= max(float(np.max(source_mass)), eps)

    endpoint_access = np.sqrt(
        source_mass[:, None] * destination_mass[None, :]
    )
    endpoint_access[~valid] = 0.0

    # Prefer edges that are both locally short and globally balanced.
    local_affinity = kernel.copy()
    local_max = float(np.max(local_affinity))
    if np.isfinite(local_max) and local_max > 0.0:
        local_affinity /= local_max

    result = (
        0.38 * balanced
        + 0.30 * cycle_support
        + 0.20 * endpoint_access
        + 0.12 * local_affinity
    )

    result[~valid] = 0.0
    np.fill_diagonal(result, 0.0)

    maximum = float(np.max(result))
    if np.isfinite(maximum) and maximum > 0.0:
        result /= maximum
    else:
        result.fill(0.0)

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(float, copy=False)
