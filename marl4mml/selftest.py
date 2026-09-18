"""Self-contained correctness checks, run with: python -m marl4mml.selftest"""
import copy

import torch

from .config import Config
from .data import BatchSampler, make_synthetic_dataset
from .gradient_control import StepBundle
from .observation import ObservationBuilder
from .policies import CentralCritic, GRUActor, coma_advantages
from .task_model import TwoModalClassifier
from .trainer import MARLMCRTrainer


def tiny_config():
    return Config(warmup_steps=2, window_size=2, windows_per_episode=3,
                  n_fit=512, n_reward=128, n_test=256, batch_size=32,
                  n_step=2, critic_epochs=2, actor_epochs=1)


def test_gradient_assembly():
    """With k1=k2=1 the assembled per-group gradients must equal the plain
    backprop gradients of the single scalar loss L0 + lambda_M (M1 + M2)."""
    cfg = tiny_config()
    torch.manual_seed(0)
    model = TwoModalClassifier(cfg.input_dim1, cfg.input_dim2, cfg.hidden_dim,
                               cfg.feat_dim, cfg.num_classes)
    ref = copy.deepcopy(model)
    fit, _, _ = make_synthetic_dataset(cfg)
    batch = BatchSampler(fit, cfg.batch_size, 0).next()

    bundle = StepBundle(model, cfg, batch, generator=torch.Generator().manual_seed(7))
    model.zero_grad()
    bundle.assemble(model, k1=1.0, k2=1.0,
                    lambda_m=cfg.lambda_m, lambda_uni=cfg.lambda_uni)

    out = ref(batch[0], batch[1])
    l_f = torch.nn.functional.cross_entropy(out["logits_f"], batch[2])
    l_1 = torch.nn.functional.cross_entropy(out["logits_1"], batch[2])
    l_2 = torch.nn.functional.cross_entropy(out["logits_2"], batch[2])
    _, _, M1, M2 = ref.mipd_terms(out["z1"], out["z2"], cfg.num_perms,
                                  torch.Generator().manual_seed(7))
    (l_f + cfg.lambda_uni * (l_1 + l_2) + cfg.lambda_m * (M1 + M2)).backward()

    max_diff = 0.0
    for p, q in zip(model.parameters(), ref.parameters()):
        assert p.grad is not None and q.grad is not None
        max_diff = max(max_diff, (p.grad - q.grad).abs().max().item())
    assert max_diff < 1e-5, f"assembled vs scalar-loss grads differ by {max_diff}"
    print(f"[ok] gradient assembly matches scalar-loss backprop (max diff {max_diff:.2e})")


def test_observation_and_policies():
    cfg = tiny_config()
    torch.manual_seed(0)
    model = TwoModalClassifier(cfg.input_dim1, cfg.input_dim2, cfg.hidden_dim,
                               cfg.feat_dim, cfg.num_classes)
    fit, _, _ = make_synthetic_dataset(cfg)
    bundle = StepBundle(model, cfg, BatchSampler(fit, cfg.batch_size, 0).next())

    ob = ObservationBuilder(cfg)
    o1 = ob.build(1, bundle.stats, -1.0, 0.0, 0, 3, 1.0)
    o2 = ob.build(2, bundle.stats, -1.0, 0.0, 0, 3, 1.0)
    assert o1.shape == (15,) and torch.isfinite(o1).all()
    assert o2.shape == (15,) and torch.isfinite(o2).all()

    actor = GRUActor(cfg.obs_dim, cfg.actor_hidden, 5)
    h = actor.init_h()
    logits, h = actor(o1.unsqueeze(0), h)
    assert logits.shape == (1, 5) and h.shape == (1, 64)

    critic = CentralCritic(2 * cfg.obs_dim + 2, cfg.critic_hidden, 25)
    hc = critic.init_h()
    q, hc = critic(ob.central(o1, o2, -1.0, -1.0).unsqueeze(0), hc)
    assert q.shape == (1, 25)
    print("[ok] observation is 15-dim finite; actor/critic output shapes correct")


def test_coma_advantage():
    Q = torch.arange(25, dtype=torch.float32)
    pi1 = torch.softmax(torch.randn(5), dim=0)
    pi2 = torch.softmax(torch.randn(5), dim=0)
    a1, a2 = 2, 3
    A1, A2 = coma_advantages(Q, pi1, pi2, a1, a2)
    Qm = Q.view(5, 5)
    assert torch.isclose(A1, Qm[a1, a2] - (pi1 * Qm[:, a2]).sum())
    assert torch.isclose(A2, Qm[a1, a2] - (pi2 * Qm[a1, :]).sum())
    print("[ok] COMA counterfactual advantages match definition")


def test_episode_and_update():
    cfg = tiny_config()
    trainer = MARLMCRTrainer(cfg)
    traj, model = trainer.run_episode(seed=0)
    assert len(traj) == cfg.windows_per_episode
    assert traj[-1]["done"] and all(torch.isfinite(torch.tensor(t["r"])) for t in traj)
    logs = trainer.rl_update([traj])
    assert all(torch.isfinite(torch.tensor(v)) for v in logs.values())

    traj_g, model_g = trainer.run_episode(seed=1, greedy=True)
    assert traj_g is not None
    metrics = trainer.test_metrics(model_g)
    assert all(0.0 <= v <= 1.0 for v in metrics.values())

    _, model_b = trainer.run_episode(seed=2, fixed_k=(-1.0, -1.0))
    print(f"[ok] episode + COMA update + greedy rollout + fixed-k baseline run "
          f"(test acc_f={metrics['acc_f']:.3f})")


if __name__ == "__main__":
    test_gradient_assembly()
    test_observation_and_policies()
    test_coma_advantage()
    test_episode_and_update()
    print("all selftests passed")
