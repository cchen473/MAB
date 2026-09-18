import copy

import torch
import torch.nn.functional as F

from .data import BatchSampler, make_synthetic_dataset
from .gradient_control import StepBundle, optimizer_step
from .observation import ObservationBuilder
from .policies import CentralCritic, GRUActor, coma_advantages
from .task_model import TwoModalClassifier


class MARLMCRTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        self.fit_set, self.reward_set, self.test_set = make_synthetic_dataset(cfg)

        n_actions = len(cfg.action_set)
        self.actors = [GRUActor(cfg.obs_dim, cfg.actor_hidden, n_actions).to(cfg.device)
                       for _ in range(2)]
        self.critic = CentralCritic(2 * cfg.obs_dim + 2, cfg.critic_hidden,
                                    n_actions ** 2).to(cfg.device)
        self.target_critic = copy.deepcopy(self.critic)
        for p in self.target_critic.parameters():
            p.requires_grad_(False)

        self.opt_actors = torch.optim.Adam(
            list(self.actors[0].parameters()) + list(self.actors[1].parameters()),
            lr=cfg.rl_lr)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=cfg.rl_lr)
        self.action_set = torch.tensor(cfg.action_set, dtype=torch.float32,
                                       device=cfg.device)
        self._update_idx = 0

    # ------------------------------------------------------------------ task
    def make_task_model(self, seed):
        torch.manual_seed(seed)
        model = TwoModalClassifier(self.cfg.input_dim1, self.cfg.input_dim2,
                                   self.cfg.hidden_dim, self.cfg.feat_dim,
                                   self.cfg.num_classes).to(self.cfg.device)
        opt = torch.optim.Adam(model.parameters(), lr=self.cfg.lr,
                               weight_decay=self.cfg.weight_decay)
        total = self.cfg.warmup_steps + self.cfg.windows_per_episode * self.cfg.window_size
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total)
        return model, opt, sched

    @torch.no_grad()
    def fusion_risk(self, model):
        model.eval()
        x1, x2, y = [t.to(self.cfg.device) for t in self.reward_set.tensors]
        risk = 0.0
        for s in range(0, len(y), 256):
            out = model(x1[s:s + 256], x2[s:s + 256])
            risk += F.cross_entropy(out["logits_f"], y[s:s + 256],
                                    reduction="sum").item()
        model.train()
        return risk / len(y)

    @torch.no_grad()
    def test_metrics(self, model):
        model.eval()
        x1, x2, y = [t.to(self.cfg.device) for t in self.test_set.tensors]
        correct = {k: 0 for k in ("f", "1", "2")}
        for s in range(0, len(y), 512):
            out = model(x1[s:s + 512], x2[s:s + 512])
            for k in correct:
                correct[k] += (out[f"logits_{k}"].argmax(1) == y[s:s + 512]).sum().item()
        model.train()
        return {f"acc_{k}": v / len(y) for k, v in correct.items()}

    # ---------------------------------------------------------------- episode
    def run_episode(self, seed, greedy=False, fixed_k=None, collect=True):
        """One episode = one full task-model training run under the controller.

        fixed_k=(k1, k2) runs a fixed-strategy baseline (no actors, no reward).
        """
        cfg = self.cfg
        model, opt, sched = self.make_task_model(seed)
        sampler = BatchSampler(self.fit_set, cfg.batch_size, seed + 1)
        ob = ObservationBuilder(cfg)
        T, W = cfg.windows_per_episode, cfg.window_size

        for _ in range(cfg.warmup_steps):
            bundle = StepBundle(model, cfg, sampler.next())
            optimizer_step(model, bundle, opt, sched, cfg.warmup_k, cfg.warmup_k, cfg)
            ob.update(bundle.stats)

        if fixed_k is not None:
            for _ in range(T * W):
                bundle = StepBundle(model, cfg, sampler.next())
                optimizer_step(model, bundle, opt, sched, fixed_k[0], fixed_k[1], cfg)
            return None, model

        h = [a.init_h(device=cfg.device) for a in self.actors]
        k_prev = (cfg.warmup_k, cfg.warmup_k)
        r_prev = 0.0
        R_prev = self.fusion_risk(model)
        traj = []

        for t in range(T):
            bundle = StepBundle(model, cfg, sampler.next())
            lr_ratio = opt.param_groups[0]["lr"] / cfg.lr
            o1 = ob.build(1, bundle.stats, k_prev[0], r_prev, t, T, lr_ratio)
            o2 = ob.build(2, bundle.stats, k_prev[1], r_prev, t, T, lr_ratio)
            central = ob.central(o1, o2, k_prev[0], k_prev[1])

            pis, acts = [], []
            with torch.no_grad():
                for i in (0, 1):
                    logits, h[i] = self.actors[i](o1.unsqueeze(0) if i == 0 else o2.unsqueeze(0), h[i])
                    pi = torch.softmax(logits, dim=-1).squeeze(0)
                    pis.append(pi)
                    acts.append(pi.argmax() if greedy else torch.distributions.Categorical(pi).sample())
            k1 = self.action_set[acts[0]].item()
            k2 = self.action_set[acts[1]].item()

            optimizer_step(model, bundle, opt, sched, k1, k2, cfg)
            for _ in range(W - 1):
                bundle = StepBundle(model, cfg, sampler.next())
                optimizer_step(model, bundle, opt, sched, k1, k2, cfg)
                ob.update(bundle.stats)

            R_new = self.fusion_risk(model)
            r = (R_prev - R_new) * cfg.reward_scale
            R_prev = R_new

            if collect:
                traj.append({
                    "obs1": o1, "obs2": o2, "central": central,
                    "a1": acts[0].detach(), "a2": acts[1].detach(),
                    "pi1": pis[0].detach(), "pi2": pis[1].detach(),
                    "r": r, "done": t == T - 1,
                })
            k_prev, r_prev = (k1, k2), r
        return traj, model

    # -------------------------------------------------------------- RL update
    def _nstep_targets(self, traj):
        """G_t = sum_{l<n} r_{t+l} + V_bar(H_{t+n}); V_bar uses the target
        critic and the behavior (old) policies. No bootstrap past episode end."""
        cfg = self.cfg
        T = len(traj)
        with torch.no_grad():
            hc = self.target_critic.init_h(device=cfg.device)
            v_bar = torch.zeros(T + 1, device=cfg.device)
            for t, tr in enumerate(traj):
                q, hc = self.target_critic(tr["central"].unsqueeze(0), hc)
                Q = q.squeeze(0).view(5, 5)
                v_bar[t] = (tr["pi1"].unsqueeze(1) * Q * tr["pi2"].unsqueeze(0)).sum()
        targets = torch.zeros(T, device=cfg.device)
        for t in range(T):
            end = min(t + cfg.n_step, T)
            targets[t] = sum(tr["r"] for tr in traj[t:end])
            if end < T:
                targets[t] += v_bar[end]
        return targets

    def rl_update(self, trajs):
        cfg = self.cfg
        beta_H = cfg.beta_H_start + (cfg.beta_H_end - cfg.beta_H_start) \
            * min(self._update_idx / max(cfg.n_updates - 1, 1), 1.0)
        logs = {"beta_H": beta_H}

        targets = [self._nstep_targets(tr) for tr in trajs]

        for _ in range(cfg.critic_epochs):
            self.opt_critic.zero_grad()
            loss_q = 0.0
            for traj, tgt in zip(trajs, targets):
                hc = self.critic.init_h(device=cfg.device)
                for t, tr in enumerate(traj):
                    q, hc = self.critic(tr["central"].unsqueeze(0), hc)
                    q_taken = q.squeeze(0)[tr["a1"] * 5 + tr["a2"]]
                    loss_q = loss_q + (q_taken - tgt[t]) ** 2
            loss_q = loss_q / sum(len(tr) for tr in trajs)
            loss_q.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.rl_grad_clip)
            self.opt_critic.step()
        logs["critic_loss"] = loss_q.item()

        for _ in range(cfg.actor_epochs):
            self.opt_actors.zero_grad()
            loss_pi, ent = 0.0, 0.0
            for traj in trajs:
                h = [a.init_h(device=cfg.device) for a in self.actors]
                with torch.no_grad():
                    hc = self.critic.init_h(device=cfg.device)
                    qs = []
                    for tr in traj:
                        q, hc = self.critic(tr["central"].unsqueeze(0), hc)
                        qs.append(q.squeeze(0))
                for t, tr in enumerate(traj):
                    pis = []
                    for i in (0, 1):
                        logits, h[i] = self.actors[i](
                            (tr["obs1"] if i == 0 else tr["obs2"]).unsqueeze(0), h[i])
                        pis.append(torch.softmax(logits, dim=-1).squeeze(0))
                    A1, A2 = coma_advantages(qs[t], pis[0], pis[1],
                                             tr["a1"], tr["a2"])
                    loss_pi = loss_pi \
                        - torch.log(pis[0][tr["a1"]] + 1e-8) * A1 \
                        - torch.log(pis[1][tr["a2"]] + 1e-8) * A2
                    ent = ent + sum(-(p * p.log()).sum() for p in pis)
            n_steps = sum(len(tr) for tr in trajs)
            loss = loss_pi / n_steps - beta_H * ent / n_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.actors[0].parameters()) + list(self.actors[1].parameters()),
                cfg.rl_grad_clip)
            self.opt_actors.step()
        logs["actor_loss"] = loss.item()

        self._update_idx += 1
        if self._update_idx % cfg.target_sync_interval == 0:
            self.target_critic.load_state_dict(self.critic.state_dict())
        return logs

    # ------------------------------------------------------------ checkpoints
    def save_controller(self, path):
        torch.save({
            "actor1": self.actors[0].state_dict(),
            "actor2": self.actors[1].state_dict(),
            "critic": self.critic.state_dict(),
            "update_idx": self._update_idx,
        }, path)

    def load_controller(self, path):
        ckpt = torch.load(path, map_location=self.cfg.device, weights_only=True)
        self.actors[0].load_state_dict(ckpt["actor1"])
        self.actors[1].load_state_dict(ckpt["actor2"])
        self.critic.load_state_dict(ckpt["critic"])
        self.target_critic.load_state_dict(ckpt["critic"])
        self._update_idx = ckpt.get("update_idx", 0)
