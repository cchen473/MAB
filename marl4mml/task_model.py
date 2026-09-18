import torch
import torch.nn as nn
import torch.nn.functional as F


def js_divergence(logits_p, logits_q):
    """Batch-mean JSD between two predictive distributions (as in MCR)."""
    p = F.softmax(logits_p, dim=1)
    q = F.softmax(logits_q, dim=1)
    m = 0.5 * (p + q)
    return 0.5 * (
        F.kl_div(m.log(), p, reduction="batchmean")
        + F.kl_div(m.log(), q, reduction="batchmean")
    )


class ModalityEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, feat_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feat_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class TwoModalClassifier(nn.Module):
    """Two encoders + concat linear fusion head + two unimodal aux heads."""

    def __init__(self, input_dim1, input_dim2, hidden_dim, feat_dim, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.enc1 = ModalityEncoder(input_dim1, hidden_dim, feat_dim)
        self.enc2 = ModalityEncoder(input_dim2, hidden_dim, feat_dim)
        self.head1 = nn.Linear(feat_dim, num_classes)
        self.head2 = nn.Linear(feat_dim, num_classes)
        self.head_f = nn.Linear(2 * feat_dim, num_classes)

    def forward(self, x1, x2):
        z1 = self.enc1(x1)
        z2 = self.enc2(x2)
        return {
            "z1": z1,
            "z2": z2,
            "logits_f": self.fusion_logits(z1, z2),
            "logits_1": self.head1(z1),
            "logits_2": self.head2(z2),
        }

    def fusion_logits(self, z1, z2):
        return self.head_f(torch.cat([z1, z2], dim=1))

    def mipd_terms(self, z1, z2, num_perms=1, generator=None):
        """Contribution proxies C_i (mean JSD under batch permutation) and M_i = -C_i."""
        B = z1.size(0)
        logits_f = self.fusion_logits(z1, z2)
        C1 = z1.new_zeros(())
        C2 = z1.new_zeros(())
        for _ in range(num_perms):
            perm1 = torch.randperm(B, device=z1.device, generator=generator)
            perm2 = torch.randperm(B, device=z2.device, generator=generator)
            C1 = C1 + js_divergence(logits_f, self.fusion_logits(z1[perm1], z2))
            C2 = C2 + js_divergence(logits_f, self.fusion_logits(z1, z2[perm2]))
        C1 = C1 / num_perms
        C2 = C2 / num_perms
        return C1, C2, -C1, -C2

    def param_groups(self):
        return {
            "enc1": list(self.enc1.parameters()),
            "enc2": list(self.enc2.parameters()),
            "head_f": list(self.head_f.parameters()),
            "head1": list(self.head1.parameters()),
            "head2": list(self.head2.parameters()),
        }
