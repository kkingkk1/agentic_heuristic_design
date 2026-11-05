import numpy as np

def heuristics_v2(edge_attr: np.ndarray) -> np.ndarray:
    x = np.asarray(edge_attr, dtype=float)

    if x.ndim != 2 or x.shape[1] != 1:
        raise ValueError("edge_attr must have shape (n_edges, 1)")

    n_edges = x.shape[0]
    if n_edges == 0:
        return np.empty(0, dtype=float)

    values = x[:, 0]
    finite = np.isfinite(values)

    if not np.any(finite):
        return np.full(n_edges, 0.5, dtype=float)

    valid = values[finite]
    median = np.median(valid)
    mad = np.median(np.abs(valid - median))
    scale = 1.4826 * mad

    if scale <= np.finfo(float).eps:
        scale = np.std(valid)
    if scale <= np.finfo(float).eps:
        scale = 1.0

    standardized = np.zeros(n_edges, dtype=float)
    standardized[finite] = (values[finite] - median) / scale
    standardized = np.clip(standardized, -12.0, 12.0)

    order = np.argsort(np.where(finite, values, np.inf), kind="mergesort")
    ranks = np.empty(n_edges, dtype=float)
    ranks[order] = np.arange(n_edges, dtype=float)

    valid_count = np.count_nonzero(finite)
    if valid_count > 1:
        percentile = 1.0 - ranks / (valid_count - 1.0)
    else:
        percentile = np.full(n_edges, 0.5, dtype=float)

    sigmoid = 1.0 / (1.0 + np.exp(standardized))
    score = 0.6 * percentile + 0.4 * sigmoid

    score[~finite] = 0.0
    return np.clip(score, 0.0, 1.0)
