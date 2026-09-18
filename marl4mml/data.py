import math

import torch
from torch.utils.data import TensorDataset


def make_synthetic_dataset(cfg):
    """Two-modality classification data with asymmetric modality strength.

    Each modality carries the class label through its own random projection;
    snr_mod controls how informative the modality is. Modality 2 is weaker by
    default, which induces the imbalance MCR-style regularization targets.
    """
    g = torch.Generator().manual_seed(cfg.seed)
    n_total = cfg.n_fit + cfg.n_reward + cfg.n_test
    K = cfg.num_classes

    y = torch.randint(0, K, (n_total,), generator=g)
    class_vec = torch.eye(K)

    W1 = torch.randn(K, cfg.input_dim1, generator=g) / math.sqrt(cfg.input_dim1)
    W2 = torch.randn(K, cfg.input_dim2, generator=g) / math.sqrt(cfg.input_dim2)

    s = class_vec[y]
    x1 = cfg.snr_mod1 * (s @ W1) * math.sqrt(cfg.input_dim1) + torch.randn(n_total, cfg.input_dim1, generator=g)
    x2 = cfg.snr_mod2 * (s @ W2) * math.sqrt(cfg.input_dim2) + torch.randn(n_total, cfg.input_dim2, generator=g)

    n1, n2 = cfg.n_fit, cfg.n_fit + cfg.n_reward
    fit = TensorDataset(x1[:n1], x2[:n1], y[:n1])
    reward = TensorDataset(x1[n1:n2], x2[n1:n2], y[n1:n2])
    test = TensorDataset(x1[n2:], x2[n2:], y[n2:])
    return fit, reward, test


class BatchSampler:
    """Infinite minibatch iterator with a fresh permutation each pass."""

    def __init__(self, dataset, batch_size, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.g = torch.Generator().manual_seed(seed)
        self._order = None
        self._pos = 0

    def next(self):
        n = len(self.dataset)
        if self._order is None or self._pos + self.batch_size > n:
            self._order = torch.randperm(n, generator=self.g)
            self._pos = 0
        idx = self._order[self._pos:self._pos + self.batch_size]
        self._pos += self.batch_size
        return tuple(t[idx] for t in self.dataset.tensors)
