import math

import torch


class ObservationBuilder:
    """Builds the 15-dim local observation per modality.

    [ l_i/logK, d(l_i), acc_i, E_i,               (prediction state, 4)
      C_i/log2, d(C_i),                            (contribution state, 2)
      n_i, s_i, q_i, chi_i, rho_i,                 (local gradient state, 5)
      k_{i,t-1}, r_{t-1}, t/T, lr_t/lr_0 ]         (own history + public, 4)

    EMA-tracked quantities are updated every minibatch; deltas are taken
    between consecutive control steps. Everything is detached scalars.
    """

    KEYS = ("loss", "acc", "entropy", "C")

    def __init__(self, cfg):
        self.cfg = cfg
        self.alpha = cfg.ema_alpha
        self.logK = math.log(cfg.num_classes)
        self.reset()

    def reset(self):
        self.ema = {i: {k: None for k in self.KEYS} for i in (1, 2)}
        self.prev_step_ema = {i: {"loss": 0.0, "C": 0.0} for i in (1, 2)}

    def update(self, stats):
        for i in (1, 2):
            for k in self.KEYS:
                v = stats[i][{"loss": "loss_uni", "acc": "acc_uni",
                              "entropy": "entropy", "C": "C"}[k]]
                e = self.ema[i][k]
                self.ema[i][k] = v if e is None else (1 - self.alpha) * e + self.alpha * v

    def build(self, i, stats, k_prev, r_prev, t, T, lr_ratio):
        self.update(stats)
        e = self.ema[i]
        s = stats[i]
        loss_bar = e["loss"] / self.logK
        c_bar = e["C"] / math.log(2)
        d_loss = loss_bar - self.prev_step_ema[i]["loss"]
        d_c = c_bar - self.prev_step_ema[i]["C"]
        self.prev_step_ema[i]["loss"] = loss_bar
        self.prev_step_ema[i]["C"] = c_bar
        obs = [
            loss_bar, d_loss, e["acc"], e["entropy"],
            c_bar, d_c,
            s["n"], s["s"], s["q"], s["chi"], s["rho"],
            k_prev, r_prev, t / max(T, 1), lr_ratio,
        ]
        return torch.tensor(obs, dtype=torch.float32, device=self.cfg.device)

    def central(self, o1, o2, k1_prev, k2_prev):
        return torch.cat([o1, o2, torch.tensor([k1_prev, k2_prev],
                                               dtype=torch.float32, device=self.cfg.device)])
