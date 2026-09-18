import torch
import torch.nn.functional as F


def _flat(ts):
    return torch.cat([t.reshape(-1) for t in ts])


def _cos(a, b, eps=1e-8):
    return (torch.dot(a, b) / (a.norm() * b.norm() + eps)).item()


class StepBundle:
    """Gradients of every loss component w.r.t. every parameter group,
    computed on one minibatch, plus detached statistics for observations."""

    def __init__(self, model, cfg, batch, generator=None):
        x1, x2, y = [t.to(cfg.device) for t in batch]
        out = model(x1, x2)
        l_f = F.cross_entropy(out["logits_f"], y)
        l_1 = F.cross_entropy(out["logits_1"], y)
        l_2 = F.cross_entropy(out["logits_2"], y)
        C1, C2, M1, M2 = model.mipd_terms(out["z1"], out["z2"], cfg.num_perms, generator)

        groups = model.param_groups()
        e1, e2 = groups["enc1"], groups["enc2"]
        hf, h1, h2 = groups["head_f"], groups["head1"], groups["head2"]

        def grads(loss, params, retain):
            return torch.autograd.grad(loss, params, retain_graph=retain, allow_unused=False)

        # split each multi-group grad call back into per-group lists
        g = grads(l_f, e1 + e2 + hf, retain=True)
        self.gf_1, self.gf_2, self.gf_hf = g[:len(e1)], g[len(e1):len(e1) + len(e2)], g[len(e1) + len(e2):]
        self.gu_1 = grads(l_1, e1, retain=True)
        self.gu_2 = grads(l_2, e2, retain=True)
        self.gh_1 = grads(l_1, h1, retain=True)
        self.gh_2 = grads(l_2, h2, retain=True)
        g = grads(M1, e1 + e2 + hf, retain=True)
        self.u_1, self.v_2, self.gM1_hf = g[:len(e1)], g[len(e1):len(e1) + len(e2)], g[len(e1) + len(e2):]
        g = grads(M2, e1 + e2 + hf, retain=False)
        self.v_1, self.u_2, self.gM2_hf = g[:len(e1)], g[len(e1):len(e1) + len(e2)], g[len(e1) + len(e2):]

        lam_u = cfg.lambda_uni
        with torch.no_grad():
            self.stats = {}
            for i, (gf, gu, u) in enumerate(((self.gf_1, self.gu_1, self.u_1),
                                             (self.gf_2, self.gu_2, self.u_2)), start=1):
                gf_v, gu_v, u_v = _flat(gf), _flat(gu), _flat(u)
                g0_v = gf_v + lam_u * gu_v
                d_i = g0_v.numel()
                p_i = F.softmax(out[f"logits_{i}"], dim=1)
                self.stats[i] = {
                    "loss_uni": l_1.item() if i == 1 else l_2.item(),
                    "acc_uni": (p_i.argmax(1) == y).float().mean().item(),
                    "entropy": (-(p_i * p_i.log()).sum(1).mean()
                                / torch.log(torch.tensor(float(model.num_classes)))).item(),
                    "C": C1.item() if i == 1 else C2.item(),
                    "n": torch.log1p(g0_v.norm() / d_i ** 0.5).item(),
                    "s": torch.log1p(gf_v.norm() / (lam_u * gu_v.norm() + 1e-8)).item(),
                    "q": torch.log1p(cfg.lambda_m * u_v.norm() / (g0_v.norm() + 1e-8)).item(),
                    "chi": _cos(gu_v, gf_v),
                    "rho": _cos(u_v, gf_v),
                }
            self.stats["loss_f"] = l_f.item()

    @torch.no_grad()
    def assemble(self, model, k1, k2, lambda_m, lambda_uni):
        """theta_i update direction: g_i^0 + lambda_M (u_i + k_i v_i);
        fusion head: grad of L0 + lambda_M (M1 + M2). Written into .grad so the
        original optimizer protocol (Adam, schedule, wd) applies unchanged."""
        groups = model.param_groups()
        k = {1: k1, 2: k2}
        for i in (1, 2):
            params = groups[f"enc{i}"]
            gf = getattr(self, f"gf_{i}")
            gu = getattr(self, f"gu_{i}")
            u = getattr(self, f"u_{i}")
            v = getattr(self, f"v_{i}")
            for p, a, b, c, d in zip(params, gf, gu, u, v):
                p.grad = a + lambda_uni * b + lambda_m * (c + k[i] * d)
        for p, a, b, c in zip(groups["head_f"], self.gf_hf, self.gM1_hf, self.gM2_hf):
            p.grad = a + lambda_m * (b + c)
        for p, a in zip(groups["head1"], self.gh_1):
            p.grad = lambda_uni * a
        for p, a in zip(groups["head2"], self.gh_2):
            p.grad = lambda_uni * a


def optimizer_step(model, bundle, optimizer, scheduler, k1, k2, cfg):
    optimizer.zero_grad(set_to_none=True)
    bundle.assemble(model, k1, k2, cfg.lambda_m, cfg.lambda_uni)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
