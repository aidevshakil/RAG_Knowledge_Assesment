"""Maximal Marginal Relevance re-ranking.

Pure similarity ranking often returns five near-duplicate chunks from the same
page. MMR picks each next chunk to maximise

    lambda * sim(query, chunk) - (1 - lambda) * max sim(chunk, already_selected)

which trades a little relevance for coverage — usually a net win for grounded
answers, because the LLM sees several angles instead of one repeated.
"""

from __future__ import annotations

import numpy as np


def mmr_select(
    query_vector: np.ndarray,
    candidate_vectors: np.ndarray,
    scores: np.ndarray,
    k: int,
    lambda_mult: float = 0.7,
) -> list[int]:
    """Indices of the `k` selected candidates, in selection order."""
    count = candidate_vectors.shape[0]
    if count == 0 or k <= 0:
        return []
    k = min(k, count)
    if lambda_mult >= 1.0:
        return list(np.argsort(-scores)[:k])

    similarity = candidate_vectors @ candidate_vectors.T
    selected: list[int] = [int(np.argmax(scores))]
    remaining = set(range(count)) - set(selected)

    while len(selected) < k and remaining:
        indices = np.fromiter(remaining, dtype=np.int64, count=len(remaining))
        redundancy = similarity[np.ix_(indices, selected)].max(axis=1)
        gain = lambda_mult * scores[indices] - (1.0 - lambda_mult) * redundancy
        best = int(indices[int(np.argmax(gain))])
        selected.append(best)
        remaining.discard(best)
    return selected
