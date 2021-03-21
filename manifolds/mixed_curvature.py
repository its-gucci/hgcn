"""Mixed Poincare Ball/Euclidean product manifold."""

import torch

from manifolds.base import Manifold
from manifolds.euclidean import Euclidean
from manifolds.poincare import PoincareBall

class MixedCurvature(Manifold):
    """
    Mixed Poincare Ball/Euclidean product manifold class.
    """

    def __init__(self, split_idx):
        super(MixedCurvature, self).__init__()
        self.name = 'MixedCurvature'
        self.idx = split_idx
        self.Euc = Euclidean()
        self.Hyp = PoincareBall()

    def sqdist(self, p1, p2, c):
        e = self.Euc.sqdist(p1[..., :self.idx], p2[..., :self.idx], c)
        h = self.Hyp.sqdist(p1[..., self.idx:], p2[..., self.idx:], c)
        return e + h

    def egrad2rgrad(self, p, dp, c):
        e = self.Euc.egrad2rgrad(p[..., :self.idx], dp[..., :self.idx], c)
        h = self.Hyp.egrad2rgrad(p[..., self.idx:], dp[..., self.idx:], c)
        return torch.cat([e, h], dim=-1)

    def proj(self, p, c):
        e = self.Euc.proj(p[..., :self.idx], c)
        h = self.Hyp.proj(p[..., self.idx:], c)
        return torch.cat([e, h], dim=-1)

    def proj_tan(self, u, p, c):
        e = self.Euc.proj_tan(u[..., :self.idx], p[..., :self.idx], c)
        h = self.Hyp.proj_tan(u[..., self.idx:], p[..., self.idx:], c)
        return torch.cat([e, h], dim=-1)

    def proj_tan0(self, u, c):
        e = self.Euc.proj_tan0(u[..., :self.idx], c)
        h = self.Hyp.proj_tan0(u[..., self.idx:], c)
        return torch.cat([e, h], dim=-1)

    def expmap(self, u, p, c):
        e = self.Euc.expmap(u[..., :self.idx], p[..., :self.idx], c)
        h = self.Hyp.expmap(u[..., self.idx:], p[..., self.idx:], c)
        return torch.cat([e, h], dim=-1)

    def logmap(self, p1, p2, c):
        e = self.Euc.logmap(p1[..., :self.idx], p2[..., :self.idx], c)
        h = self.Hyp.sqdist(p1[..., self.idx:], p2[..., self.idx:], c)
        return torch.cat([e, h], dim=-1)

    def expmap0(self, u, c):
        e = self.Euc.expmap0(u[..., :self.idx], c)
        h = self.Hyp.expmap0(u[..., self.idx:], c)
        return torch.cat([e, h], dim=-1)

    def logmap0(self, p, c):
        e = self.Euc.proj(p[..., :self.idx], c)
        h = self.Hyp.proj(p[..., self.idx:], c)
        return torch.cat([e, h], dim=-1)

    def mobius_add(self, x, y, c, dim=-1):
        e = self.Euc.mobius_add(x[..., :self.idx], y[..., :self.idx], c, dim)
        h = self.Hyp.mobius_add(x[..., self.idx:], y[..., self.idx:], c, dim)
        return torch.cat([e, h], dim=-1)

    def mobius_matvec(self, m, x, c):
        e = self.Euc.mobius_matvec(m, x[:self.idx], c)
        h = self.Hyp.mobius_matvec(m, x[self.idx:], c)
        return torch.cat([e, h], dim=0)

    def init_weights(self, w, c, irange=1e-5):
        w.data.uniform_(-irange, irange)
        return w

    def inner(self, p, c, u, v=None, keepdim=False):
        e = self.Euc.inner(p[..., :self.idx], c, u[..., :self.idx], v=v[..., :self.idx], keepdim=keepdim)
        h = self.Hyp.inner(p[..., self.idx:], c, u[..., self.idx:], v=v[..., self.idx:], keepdim=keepdim)
        return e + h

    def ptransp(self, x, y, v, c):
        e = self.Euc.ptransp(x[..., :self.idx], y[..., :self.idx], v[..., :self.idx], c)
        h = self.Hyp.ptransp(x[..., self.idx:], y[..., self.idx:], v[..., self.idx:], c)
        return torch.cat([e, h], dim=-1)

    def ptransp0(self, x, v, c):
        e = self.Euc.ptransp0(x[:self.idx], v[:self.idx], c)
        h = self.Hyp.ptransp0(x[self.idx:], v[self.idx:], c)
        return torch.cat([e, h], dim=-1)

