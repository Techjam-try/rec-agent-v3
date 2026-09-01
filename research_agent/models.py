"""Numpy models: FM pointwise, pairwise BPR, and shared-representation MTL."""
from __future__ import annotations

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class FM:
    def __init__(self, dim, k=16, lr=0.001, l2=1e-6, seed=0, heads=1):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, .01, (dim, k)).astype(np.float32)
        self.W = np.zeros((heads, dim), dtype=np.float32)
        self.b = np.zeros(heads, dtype=np.float32)
        self.lr, self.l2, self.heads, self.t = lr, l2, heads, 0
        self.mV = np.zeros_like(self.V); self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)

    def _parts(self, X):
        E = self.V[X]; S = E.sum(1)
        interaction = .5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        return E, S, interaction

    def logits(self, X):
        _, _, inter = self._parts(X)
        return inter[:, None] + self.W[:, X].sum(2).T + self.b

    def predict(self, X, bs=200_000):
        return np.concatenate([self.logits(X[i:i + bs])[:, 0] for i in range(0, len(X), bs)])

    def _apply(self, X, grad):
        # grad has one column per task head and is d(loss)/d(logit).
        E, S, _ = self._parts(X)
        g_inter = grad.sum(1).astype(np.float32)
        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)
        np.add.at(gV, X, g_inter[:, None, None] * (S[:, None, :] - E))
        for h in range(self.heads):
            np.add.at(gW[h], X, grad[:, h, None])
        gV += self.l2 * self.V; gW += self.l2 * self.W
        self.t += 1; b1, b2, eps = .9, .999, 1e-8
        for p, g, m, v in ((self.V, gV, self.mV, self.vV), (self.W, gW, self.mW, self.vW)):
            m *= b1; m += (1-b1)*g; v *= b2; v += (1-b2)*(g*g)
            p -= self.lr * (m/(1-b1**self.t)) / (np.sqrt(v/(1-b2**self.t)) + eps)
        self.b -= self.lr * grad.sum(0)

    def step_pointwise(self, X, y, aux=None, aux_weight=.15):
        z = self.logits(X); B = len(X)
        grad = np.zeros_like(z)
        grad[:, 0] = (sigmoid(z[:, 0]) - y) / B
        loss = -np.mean(y*np.log(sigmoid(z[:, 0])+1e-9)+(1-y)*np.log(1-sigmoid(z[:, 0])+1e-9))
        if aux:
            for h, target in enumerate(aux, 1):
                # binary engagement tasks use BCE; watch-ratio uses bounded MSE.
                if h == self.heads - 1 and target.dtype.kind == "f":
                    pred = sigmoid(z[:, h]); grad[:, h] = aux_weight * 2*(pred-target)*pred*(1-pred)/B
                else:
                    grad[:, h] = aux_weight * (sigmoid(z[:, h])-target)/B
        self._apply(X, grad)
        return float(loss)

    def step_bpr(self, xp, xn):
        B = len(xp)
        zp, zn = self.logits(xp)[:, 0], self.logits(xn)[:, 0]
        g = -sigmoid(-(zp-zn)) / B
        gp = np.zeros((B, self.heads), np.float32); gn = np.zeros_like(gp)
        gp[:, 0] = g; gn[:, 0] = -g
        self._apply(xp, gp); self._apply(xn, gn)
        return float(-np.mean(np.log(sigmoid(zp-zn)+1e-9)))

    def state_dict(self):
        return {"V": self.V, "W": self.W, "b": self.b}

    def load_state_dict(self, state):
        self.V, self.W, self.b = state["V"].copy(), state["W"].copy(), state["b"].copy()


def make_bpr_pairs(users, labels, rng, limit=None):
    """One negative per positive; all indices stay inside the training split."""
    groups = {}
    for i, user in enumerate(users):
        bucket = groups.setdefault(user, [[], []])
        bucket[int(labels[i] > 0)].append(i)
    pos, neg = [], []
    for positives, negatives in groups.values():
        if positives and negatives:
            take = positives if limit is None else positives[:limit]
            pos.extend(take); neg.extend(rng.choice(negatives, len(take)).tolist())
    return np.asarray(pos, np.int32), np.asarray(neg, np.int32)
