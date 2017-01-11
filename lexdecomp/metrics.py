"""Defines evaluation metrics.
"""
import numpy as np


def mean_reciprocal_rank(labels, scores):
    # TODO docstring
    pos_correct = np.argmax(labels)
    score = scores[pos_correct]
    rank = sum(1 for s in scores if s > score)
    return 1 / (rank + 1)
