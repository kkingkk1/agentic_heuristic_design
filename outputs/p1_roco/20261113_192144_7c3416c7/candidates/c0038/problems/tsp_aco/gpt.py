import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return global assignment-based edge priors for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    valid = np.isfinite(distances)
    offdiag = ~np.eye(n, dtype=bool)
    usable = valid & offdiag

    if not np.any(usable):
        return np.zeros((n, n), dtype=float)

    clean = np.asarray(distances, dtype=float).copy()
    finite_values = clean[usable]
    finite_values = finite_values[finite_values >= 0.0]

    if finite_values.size == 0:
        return np.zeros((n, n), dtype=float)

    positive = finite_values[finite_values > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    maximum = float(np.max(finite_values))
    invalid_value = maximum + 20.0 * scale
    if not np.isfinite(invalid_value):
        invalid_value = scale

    clean[~valid] = invalid_value
    clean = np.maximum(clean, 0.0)
    np.fill_diagonal(clean, invalid_value)

    finite_offdiag = clean[usable]
    lower = float(np.min(finite_offdiag))
    shifted = np.maximum(clean - lower, 0.0)

    temperatures = scale * np.array([0.20, 0.45, 0.90, 1.80], dtype=float)
    accumulated = np.zeros((n, n), dtype=float)

    for temperature in temperatures:
        temperature = max(float(temperature), np.finfo(float).eps)

        logits = -shifted / temperature
        logits = np.clip(logits, -80.0, 0.0)

        kernel = np.exp(logits)
        kernel[~usable] = 0.0

        row_scale = np.maximum(kernel.sum(axis=1, keepdims=True), 1e-300)
        kernel /= row_scale

        for _ in range(80):
            column_sum = kernel.sum(axis=0, keepdims=True)
            kernel /= np.maximum(column_sum, 1e-300)

            row_sum = kernel.sum(axis=1, keepdims=True)
            kernel /= np.maximum(row_sum, 1e-300)

        accumulated += kernel

    assignment = accumulated / float(temperatures.size)
    assignment[~usable] = 0.0

    outgoing_mass = assignment.sum(axis=1, keepdims=True)
    incoming_mass = assignment.sum(axis=0, keepdims=True)

    outgoing = assignment / np.maximum(outgoing_mass, 1e-300)
    incoming = assignment / np.maximum(incoming_mass, 1e-300)

    two_step = outgoing @ incoming
    two_step[~usable] = 0.0

    result = assignment * (0.65 + 0.35 * np.sqrt(np.maximum(two_step, 0.0)))
    result[~usable] = 0.0
    np.fill_diagonal(result, 0.0)

    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    maximum_result = float(np.max(result))

    if maximum_result > 0.0:
        result /= maximum_result

    return result.astype(float, copy=False)
