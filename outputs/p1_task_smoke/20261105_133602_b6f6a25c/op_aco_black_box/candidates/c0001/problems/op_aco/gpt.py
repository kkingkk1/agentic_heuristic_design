import numpy as np


def heuristics_v2(
    node_attr: np.ndarray,
    edge_attr: np.ndarray,
    node_constraint: float
) -> np.ndarray:
    edge = np.asarray(edge_attr, dtype=float)
    nodes = np.asarray(node_attr, dtype=float)

    if edge.ndim != 2 or edge.shape[0] != edge.shape[1]:
        raise ValueError("edge_attr must be a square matrix")

    n = edge.shape[0]
    if nodes.ndim != 1 or nodes.shape[0] != n:
        raise ValueError("node_attr must have shape (n,)")

    if n == 0:
        return np.zeros_like(edge, dtype=float)

    valid = np.isfinite(edge)
    valid &= ~np.eye(n, dtype=bool)

    if not np.any(valid):
        return np.zeros_like(edge, dtype=float)

    safe_edge = np.where(valid, edge, np.nan)
    finite_values = safe_edge[np.isfinite(safe_edge)]

    center = np.median(finite_values)
    spread = np.median(np.abs(finite_values - center))
    spread = max(float(spread), np.finfo(float).eps)

    # Lower edge values are treated as cheaper and therefore more attractive.
    cost_score = -(edge - center) / spread
    cost_score = np.where(valid, cost_score, 0.0)

    # Prefer edges whose cost is compatible with the available total budget.
    budget = float(node_constraint)
    if np.isfinite(budget) and budget > 0.0:
        budget_scale = max(abs(budget) / max(n - 1, 1), spread)
        budget_score = -np.abs(edge - budget / max(n - 1, 1)) / budget_scale
    else:
        budget_score = np.zeros_like(edge)

    # Endpoint accessibility: nodes with several inexpensive outgoing/incoming
    # edges receive a mild bonus, encouraging structurally viable solutions.
    positive_cost = np.where(valid, np.maximum(edge - np.nanmin(finite_values), 0.0), np.nan)
    accessibility = np.nanmedian(positive_cost, axis=1)
    accessibility = np.nan_to_num(accessibility, nan=center, posinf=center, neginf=center)
    access_scale = max(
        float(np.median(np.abs(accessibility - np.median(accessibility)))),
        np.finfo(float).eps
    )
    access_score = -(
        accessibility[:, None] + accessibility[None, :] - 2.0 * np.median(accessibility)
    ) / access_scale

    # Special-node relevance: edges touching node 0 get a small prior bonus,
    # with stronger preference for the cheaper incident edge.
    special = np.zeros((n, n), dtype=float)
    if n > 1:
        special[0, 1:] = 1.0
        special[1:, 0] = 1.0
    special_score = special * (0.25 + 0.25 * cost_score)

    raw = (
        1.00 * cost_score
        + 0.35 * budget_score
        + 0.20 * access_score
        + special_score
    )
    raw = np.where(valid, raw, -np.inf)

    finite_raw = raw[np.isfinite(raw)]
    temperature = max(
        float(np.median(np.abs(finite_raw - np.median(finite_raw)))),
        0.5
    )

    shifted = np.where(np.isfinite(raw), (raw - np.max(finite_raw)) / temperature, -np.inf)
    weights = np.where(np.isfinite(shifted), np.exp(np.clip(shifted, -60.0, 0.0)), 0.0)

    row_totals = weights.sum(axis=1, keepdims=True)
    global_total = weights.sum()

    priors = np.divide(
        weights,
        row_totals,
        out=np.zeros_like(weights),
        where=row_totals > 0.0
    )

    if global_total > 0.0:
        priors = 0.5 * priors + 0.5 * (weights / global_total)

    priors[~valid] = 0.0
    return np.asarray(priors, dtype=float)
