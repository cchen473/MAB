"""MARL-controlled MCR modality-interaction training.

Stages:
  python train.py selftest                 # correctness checks
  python train.py controller               # stages 1-3: train actors + central critic
  python train.py final                    # stage 4: frozen actors, greedy actions, test
  python train.py baseline --k -1          # fixed-k strategy (MCR-style) for comparison
"""
import argparse
import os

import torch

from marl4mml.config import Config
from marl4mml.trainer import MARLMCRTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["selftest", "controller", "final", "baseline"])
    parser.add_argument("--config", default=None, help="path to config JSON")
    parser.add_argument("--out", default="runs/default")
    parser.add_argument("--device", default=None)
    parser.add_argument("--n-updates", type=int, default=None)
    parser.add_argument("--controller", default=None, help="controller checkpoint for final")
    parser.add_argument("--k", type=float, default=-1.0, help="fixed k for baseline")
    args = parser.parse_args()

    if args.stage == "selftest":
        from marl4mml.selftest import (test_coma_advantage, test_episode_and_update,
                                       test_gradient_assembly,
                                       test_observation_and_policies)
        test_gradient_assembly()
        test_observation_and_policies()
        test_coma_advantage()
        test_episode_and_update()
        print("all selftests passed")
        return

    os.makedirs(args.out, exist_ok=True)
    cfg = Config.load(args.config) if args.config else Config()
    if args.device:
        cfg.device = args.device
    if args.n_updates is not None:
        cfg.n_updates = args.n_updates
    cfg.save(os.path.join(args.out, "config.json"))

    trainer = MARLMCRTrainer(cfg)

    if args.stage == "controller":
        for update in range(cfg.n_updates):
            trajs, models = [], []
            for ep in range(cfg.episodes_per_update):
                traj, model = trainer.run_episode(seed=cfg.seed + 1000 * update + ep)
                trajs.append(traj)
                models.append(model)
            logs = trainer.rl_update(trajs)

            ret = sum(t["r"] for t in trajs[0])
            risk_T = trainer.fusion_risk(models[-1])
            metrics = trainer.test_metrics(models[-1])
            hist = torch.bincount(torch.stack([t["a1"] for t in trajs[0]]), minlength=5).tolist()
            hist2 = torch.bincount(torch.stack([t["a2"] for t in trajs[0]]), minlength=5).tolist()
            print(f"update {update:03d} | return {ret:+.4f} | R(T) {risk_T:.4f} | "
                  f"test acc_f {metrics['acc_f']:.3f} (1: {metrics['acc_1']:.3f}, "
                  f"2: {metrics['acc_2']:.3f}) | a1 {hist} a2 {hist2} | "
                  f"L_Q {logs['critic_loss']:.4f} L_pi {logs['actor_loss']:.4f} "
                  f"beta_H {logs['beta_H']:.4f}")
            trainer.save_controller(os.path.join(args.out, "controller.pt"))
        print(f"controller saved to {args.out}/controller.pt")

    elif args.stage == "final":
        ckpt = args.controller or os.path.join(args.out, "controller.pt")
        trainer.load_controller(ckpt)
        for p in trainer.actors[0].parameters():
            p.requires_grad_(False)
        for p in trainer.actors[1].parameters():
            p.requires_grad_(False)
        _, model = trainer.run_episode(seed=cfg.seed + 777, greedy=True, collect=False)
        metrics = trainer.test_metrics(model)
        print(f"[final] greedy controller | test acc_f {metrics['acc_f']:.4f} "
              f"| acc_1 {metrics['acc_1']:.4f} | acc_2 {metrics['acc_2']:.4f}")

    elif args.stage == "baseline":
        _, model = trainer.run_episode(seed=cfg.seed + 555, fixed_k=(args.k, args.k))
        metrics = trainer.test_metrics(model)
        print(f"[baseline k={args.k:+.1f}] test acc_f {metrics['acc_f']:.4f} "
              f"| acc_1 {metrics['acc_1']:.4f} | acc_2 {metrics['acc_2']:.4f}")


if __name__ == "__main__":
    main()
