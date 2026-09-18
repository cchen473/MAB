import json
from dataclasses import dataclass, field, asdict


@dataclass
class Config:
    seed: int = 0
    device: str = "cpu"

    # synthetic data
    n_fit: int = 4096
    n_reward: int = 512
    n_test: int = 2048
    input_dim1: int = 40
    input_dim2: int = 40
    num_classes: int = 4
    snr_mod1: float = 0.45
    snr_mod2: float = 0.15
    batch_size: int = 64

    # task model
    feat_dim: int = 32
    hidden_dim: int = 64

    # task losses / optimizer
    lambda_uni: float = 1.0
    lambda_m: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 0.0
    num_perms: int = 1

    # episode structure
    warmup_steps: int = 50
    warmup_k: float = -1.0
    window_size: int = 10
    windows_per_episode: int = 30

    # RL
    action_set: tuple = (-1.0, -0.5, 0.0, 0.5, 1.0)
    obs_dim: int = 15
    actor_hidden: int = 64
    critic_hidden: int = 128
    gamma: float = 1.0
    n_step: int = 5
    rl_lr: float = 3e-4
    critic_epochs: int = 4
    actor_epochs: int = 2
    beta_H_start: float = 0.01
    beta_H_end: float = 0.0
    target_sync_interval: int = 10
    ema_alpha: float = 0.1
    rl_grad_clip: float = 10.0
    reward_scale: float = 1.0

    # controller training
    n_updates: int = 60
    episodes_per_update: int = 1

    def save(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @staticmethod
    def load(path):
        with open(path) as f:
            d = json.load(f)
        if isinstance(d.get("action_set"), list):
            d["action_set"] = tuple(d["action_set"])
        return Config(**d)
