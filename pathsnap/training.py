"""Training examples for the learned transition component."""
import numpy as np


def transition_examples(raw, labels, rng):
    target = np.where(labels.any(axis=1), labels.argmax(axis=1), -1)
    valid = np.flatnonzero((target[:-1] >= 0) & (target[1:] >= 0) & ~raw["reset"])
    previous, current = target[valid], target[valid + 1]
    k = labels.shape[1]
    negatives_prev = rng.integers(k, size=(len(valid), 8))
    negatives_curr = rng.integers(k, size=(len(valid), 8))
    negatives_prev[:, :3] = previous[:, None]
    negatives_curr[:, 3:6] = current[:, None]
    distinct = (negatives_prev != previous[:, None]) | (negatives_curr != current[:, None])
    transitions = raw["transition"]
    positives = transitions[valid, previous, current]
    negatives = transitions[valid[:, None], negatives_prev, negatives_curr][distinct]
    return (np.concatenate((positives, negatives)),
            np.r_[np.ones(len(positives), dtype=bool), np.zeros(len(negatives), dtype=bool)])
