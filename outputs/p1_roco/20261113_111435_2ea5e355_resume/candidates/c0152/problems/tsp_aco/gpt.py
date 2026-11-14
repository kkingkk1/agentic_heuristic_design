import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a globally degree-balanced stochastic prior for TSP edges."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = raw.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    valid_directed = np.isfinite(raw) & (raw >= 0.0)
    np.fill_diagonal(valid_directed, False)

    if not np.any(valid_directed):
        return result

    transposed_valid = valid_directed.T
    both = valid_directed & transposed_valid
    only_forward = valid_directed & ~transposed_valid
    only_backward = ~valid_directed & transposed_valid

    distance = np.full((n, n), np.inf, dtype=float)
    distance[both] = 0.5 * (raw[both] + raw.T[both])
    distance[only_forward] = raw[only_forward]
    distance[only_backward] = raw.T[only_backward]
    np.fill_diagonal(distance, np.inf)

    valid = np.isfinite(distance)
    if not np.any(valid):
        return result

    positive = distance[valid & (distance > 0.0)]
    if positive.size:
        scale = float(np.median(positive))
    else:
        finite_values = distance[valid]
        scale = float(np.median(finite_values)) if finite_values.size else 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    with np.errstate(over="ignore", invalid="ignore"):
        kernel = np.exp(
            -np.clip(
                np.divide(
                    distance,
                    scale,
                    out=np.full_like(distance, np.inf),
                    where=valid,
                ),
                0.0,
                50.0,
            )
        )

    kernel[~valid] = 0.0
    np.fill_diagonal(kernel, 0.0)

    if not np.any(kernel > 0.0):
        return result

    # Symmetric iterative scaling approximating a maximum-entropy
    # degree-balanced edge distribution.
    factors = np.ones(n, dtype=float)
    for _ in range(24):
        row_mass = kernel @ factors
        update = np.divide(
            1.0,
            row_mass,
            out=np.ones_like(row_mass),
            where=row_mass > 0.0,
        )
        update = np.clip(update, 1e-12, 1e12)
        factors = np.sqrt(factors * update)

    balanced = kernel * factors[:, None] * factors[None, :]
    balanced[~valid] = 0.0
    np.fill_diagonal(balanced, 0.0)

    # Reward edges participating in short two-step closures, while keeping
    # this effect weaker than the globally balanced edge mass.
    closure = balanced @ balanced
    closure[~valid] = 0.0
    np.fill_diagonal(closure, 0.0)

    closure_max = float(np.max(closure))
    if closure_max > 0.0 and np.isfinite(closure_max):
        closure /= closure_max
    else:
        closure.fill(0.0)

    score = balanced * (0.85 + 0.15 * closure)
    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    score = np.maximum(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
