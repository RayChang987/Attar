"""Metric kernels for the Evaluator. Inputs are pin positions grouped by net (CSR order); all tensorized.

Wirelength models (all exact for what they define):
  hpwl      half-perimeter of the pin bounding box
  star      sum of L1 distances to the net's center of mass (a valid rectilinear Steiner tree through one point)
  stt       single-trunk Steiner tree: a trunk along the median x (or y), branches to every pin; best of the two
  rmst      rectilinear minimum spanning tree (batched Prim per degree bucket)
  steiner   RSMT estimate = HPWL for degree <= 3 (exact: the RSMT of <= 3 pins equals the HPWL), else the shortest
            of stt, rmst (degree <= RMST_MAX_DEGREE) and star: an upper bound on the RSMT built from real trees.
"""
import torch

RMST_BUCKETS = ((4, 4), (5, 8), (9, 16), (17, 32), (33, 64), (65, 128), (129, 256))
RMST_CHUNK = 32_000_000   # max distance-matrix entries per batch (memory bound)
RMST_MAX_DEGREE = RMST_BUCKETS[-1][1]


def net_bbox(p, net, E):
    lo = torch.full((E, 2), float('inf'), device=p.device).scatter_reduce(0, net[:, None].expand(-1, 2), p, 'amin')
    hi = torch.full((E, 2), -float('inf'), device=p.device).scatter_reduce(0, net[:, None].expand(-1, 2), p, 'amax')
    return lo, hi


def star(p, net, deg, E):
    c = torch.zeros(E, 2, device=p.device).index_add_(0, net, p) / deg.clamp(min=1)[:, None]
    return torch.zeros(E, device=p.device).index_add_(0, net, (p - c[net]).abs().sum(1))


def _sorted_within_nets(v, net, ptr):
    """Values of v sorted inside each net (pins of a net stay in its CSR range)."""
    lo, hi = v.min(), v.max()
    key = net.double() + 0.999 * ((v - lo) / (hi - lo + 1e-12)).double()
    return v[torch.argsort(key)]


def stt(p, net, deg, ptr, E, lo, hi):
    mid = ptr[:-1] + (deg - 1).clamp(min=0) // 2
    out = []
    for a in (0, 1):                                  # trunk along axis a at the median of the other axis
        b = 1 - a
        sv = _sorted_within_nets(p[:, b], net, ptr)
        med = sv[mid.clamp(max=len(sv) - 1)]
        branch = torch.zeros(E, device=p.device).index_add_(0, net, (p[:, b] - med[net]).abs())
        out.append((hi[:, a] - lo[:, a]) + branch)
    return torch.minimum(out[0], out[1])


def rmst(p, deg, ptr, E):
    """RMST length per net for degrees in RMST_BUCKETS (others: +inf). Batched Prim, one bucket at a time."""
    out = torch.full((E,), float('inf'), device=p.device)
    for dlo, dhi in RMST_BUCKETS:
        allnets = torch.nonzero((deg >= dlo) & (deg <= dhi)).flatten()
        D = dhi
        step = max(1, RMST_CHUNK // (D * D))
        for c0 in range(0, len(allnets), step):
            nets = allnets[c0:c0 + step]
            out[nets] = _prim(p, nets, deg, ptr, D)
    return out


def _prim(p, nets, deg, ptr, D):
    """Batched Prim: RMST length of each net in nets, pins padded to D with the first pin."""
    k = torch.arange(D, device=p.device)
    # pad with the net's first pin (zero-length duplicates do not change the MST)
    idx = ptr[nets][:, None] + torch.where(k[None, :] < deg[nets][:, None], k[None, :], torch.zeros_like(k)[None, :])
    q = p[idx]                                                     # (n, D, 2)
    dist = (q[:, :, None, :] - q[:, None, :, :]).abs().sum(-1)     # (n, D, D)
    n = len(nets)
    intree = torch.zeros(n, D, dtype=torch.bool, device=p.device); intree[:, 0] = True
    key = dist[:, 0, :].clone(); key[intree] = float('inf')
    tot = torch.zeros(n, device=p.device)
    rows = torch.arange(n, device=p.device)
    for _ in range(D - 1):
        j = key.argmin(1)
        tot += key[rows, j]
        intree[rows, j] = True
        key = torch.minimum(key, dist[rows, j, :])
        key = key.masked_fill(intree, float('inf'))
    return tot


def steiner(p, net, deg, ptr, E, lo, hi):
    hp = (hi - lo).sum(1)
    s = torch.minimum(stt(p, net, deg, ptr, E, lo, hi), star(p, net, deg, E))
    s = torch.minimum(s, rmst(p, deg, ptr, E))
    return torch.where(deg <= 3, hp, s)


def wa_value_grad(p, net, E, gamma, w):
    """Weighted-average wirelength and its analytic gradient with respect to the pins (no autograd).
    p (P, 2) pin positions grouped by net; w (E,) net weights (0 for nets with < 2 pins). Returns (value, dp)."""
    pp = torch.cat([p, -p], 1)                                                    # (P, 4): x, y, -x, -y
    m = torch.full((E, 4), -float('inf'), device=p.device).scatter_reduce(0, net[:, None].expand(-1, 4), pp, 'amax')
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    e = torch.exp((pp - m[net]) / gamma)                                          # (P, 4)
    S = torch.zeros(E, 8, device=p.device).index_add_(0, net, torch.cat([e, pp * e], 1))
    Se, Sxe = S[:, :4].clamp(min=1e-30), S[:, 4:]
    wa = Sxe / Se                                                                 # soft max of x, y, -x, -y
    val = (wa.sum(1) * w).sum()
    g = e / Se[net] * (1 + (pp - wa[net]) / gamma) * w[net][:, None]              # d wa / d pp
    return val, g[:, :2] - g[:, 2:]
