# coding=utf-8
import random
import torch
from torch.optim.optimizer import Optimizer

# --------- Helpers (kept as in your file, minor safety tweaks) ---------
def norm(v, dim: int = 1):
    assert v.dim() == 2
    return v.norm(p=2, dim=dim, keepdim=True)

def unit(v, dim: int = 1, eps: float = 1e-8):
    vnorm = norm(v, dim)
    return v / vnorm.add(eps), vnorm

def matrix_norm_one(W: torch.Tensor) -> torch.Tensor:
    # returns scalar tensor in same dtype/device
    return torch.abs(W).sum(dim=0).max()

def Cayley_loop(X, W, tan_vec, t):
    # X: [n, p], W: [k, k], tan_vec: [n, p] or [k, n]? (use your original shapes)
    n, p = X.size()
    Y = X + t * tan_vec
    for _ in range(5):
        Y = X + t * torch.matmul(W, 0.5 * (X + Y))
    return Y.t()

def qr_retraction(tan_vec):  # tan_vec: p-by-n, p <= n
    p, n = tan_vec.size()
    tan_vec_t = tan_vec.t().contiguous()
    dtype = tan_vec_t.dtype
    tan_vec_t = tan_vec_t.to(dtype=torch.float32)
    q, r = torch.linalg.qr(tan_vec_t)
    d = torch.diag(r, 0)
    ph = d.sign()
    q = q * ph.expand_as(q)
    return q.t().contiguous().to(dtype=dtype)

_EPS = 1e-8

# --------- Optimizer ---------
class SGDG(Optimizer):
    r"""
    SGD-G on Stiefel manifold (Cayley transform) or plain SGD.

    Args:
        params: iterable of parameters or param groups
        lr (float)
        momentum (float, default 0.0)
        dampening (float, default 0.0)
        weight_decay (float, default 0.0)
        nesterov (bool, default False)
        stiefel (bool, default False)
        omega (float, default 0.0)  # kept for compatibility
        grad_clip (float|None, default None)
    """

    def __init__(
        self,
        params,
        lr: float,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        stiefel: bool = False,
        omega: float = 0.0,
        grad_clip=None,
    ):
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires momentum > 0 and dampening == 0")
        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            stiefel=stiefel,
            omega=omega,
            grad_clip=grad_clip,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            momentum     = group["momentum"]
            dampening    = group["dampening"]
            weight_decay = group["weight_decay"]
            nesterov     = group["nesterov"]
            stiefel      = group["stiefel"]
            lr           = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                # Common dtype/device
                p_dtype = p.dtype
                p_dev   = p.device

                # Reshape param & grad
                P = p.view(p.size(0), -1)                      # [m, k]
                g = p.grad.view(p.size(0), -1).to(p_dtype)     # [m, k]

                # Optional grad clip
                if group["grad_clip"] is not None:
                    torch.nn.utils.clip_grad_norm_([p], group["grad_clip"])

                # Build "unity" (Q) on same dtype/device as p
                Q, _ = unit(P)                                  # [m, k]
                Q = Q.to(dtype=p_dtype, device=p_dev)

                # Occasional QR retraction for numerical hygiene
                if stiefel and Q.size(0) <= Q.size(1):
                    if random.randint(1, 101) == 1:
                        Q = qr_retraction(Q).to(dtype=p_dtype, device=p_dev)

                    # Momentum buffer V with shape matching g.t(): [k, m]
                    state = self.state[p]
                    V = state.get("momentum_buffer", None)
                    if V is None:
                        V = torch.zeros_like(g.t(), dtype=p_dtype, device=p_dev)  # [k, m]
                        state["momentum_buffer"] = V

                    # Update momentum on the manifold branch
                    V.mul_(momentum).add_(-g.t())  # V = m*V - g^T

                    # All matmuls must align in dtype/device
                    # MX: [k, m] @ [m, k] -> [k, k]
                    MX   = torch.mm(V, Q)
                    XMX  = torch.mm(Q, MX)           # [m, k]
                    XXMX = torch.mm(Q.t(), XMX)       # [k, k]
                    W_hat = MX - 0.5 * XXMX
                    W = W_hat - W_hat.t()

                    # Step size
                    # matrix_norm_one returns scalar tensor
                    denom = matrix_norm_one(W).clamp_min(_EPS)
                    t = 1.0 / denom.item()          # scalar float
                    alpha = min(1.0 * t, float(lr)) # your original had 0.5*2

                    # Cayley update (expects shapes consistent with your functions)
                    p_new = Cayley_loop(Q.t(), W, V, alpha)  # should return [k, m] -> reshape
                    V_new = torch.mm(W, Q.t())               # [k, m]

                    p.copy_(p_new.view_as(p))
                    V.copy_(V_new)

                else:
                    # Euclidean (plain SGD) branch
                    d_p = g.clone()

                    if weight_decay != 0.0:
                        d_p.add_(weight_decay, P)

                    state = self.state[p]
                    buf = state.get("momentum_buffer", None)
                    if momentum != 0.0:
                        if buf is None:
                            buf = d_p.clone()
                            state["momentum_buffer"] = buf
                        else:
                            buf.mul_(momentum).add_(1.0 - dampening, d_p)
                        d_p = d_p.add(momentum, buf) if nesterov else buf

                    p.add_(-lr, d_p.view_as(P))

        return loss