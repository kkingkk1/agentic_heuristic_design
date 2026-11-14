import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    valid = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(valid, False)

    positive_values = distances[valid & (distances > 0.0)]
    if positive_values.size:
        temperature = float(np.median(positive_values))
    else:
        temperature = 1.0
    temperature = max(temperature, np.finfo(float).eps)

    finite_valid = distances[valid]
    if finite_valid.size:
        invalid_value = max(float(np.max(finite_valid)), temperature) + temperature
    else:
        invalid_value = temperature

    safe = np.where(valid, distances, invalid_value)
    kernel = np.exp(-np.clip(safe / temperature, 0.0, 50.0))
    kernel[~valid] = 0.0
    np.fill_diagonal(kernel, 0.0)

    row_mass = np.sum(kernel, axis=1, keepdims=True)
    transition = np.divide(
        kernel,
        row_mass,
        out=np.zeros_like(kernel),
        where=row_mass > 0.0,
    )

    isolated = np.flatnonzero(row_mass[:, 0] <= 0.0)
    if isolated.size:
        transition[isolated, :] = 1.0 / float(n - 1)
        transition[isolated, isolated] = 0.0

    two_step = transition @ transition
    three_step = two_step @ transition

    direct = 0.5 * (transition + transition.T)
    diffusion_two = 0.5 * (two_step + two_step.T)
    diffusion_three = 0.5 * (three_step + three_step.T)

    score = (
        0.15 * direct
        + 0.35 * diffusion_two
        + 0.50 * diffusion_three
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
