"""Tensor kernels shared by the Evaluator and the Naive engines. Pure PyTorch, exact, no Python loops over objects.

Rectangles on a bin grid: the exact overlap of a rectangle [x0, x1] x [y0, y1] with bin (i, j) factorises as
ox_i * oy_j, and ox = g(x0) - g(x1) where g(c)_i = clamp(X_{i+1} - c, 0, bw) is a cumulative sum of a sequence
with two non-zeros. So a whole map of overlaps is: scatter 4 corners x 4 entries per rectangle into an
(nx+1, ny+1) array, then one 2D cumulative sum. Integrating a field over rectangles is the transpose: suffix-sum
the field once, gather the same 16 entries per rectangle. Both are O(#rects + #bins).
"""
import torch


class Grid:
    """A uniform bin grid: origin (x0, y0), bin size (bw, bh), nx x ny bins."""

    def __init__(self, x0, y0, x1, y1, nx, ny, device):
        self.x0, self.y0 = float(x0), float(y0)
        self.nx, self.ny = int(nx), int(ny)
        self.bw, self.bh = (float(x1) - self.x0) / self.nx, (float(y1) - self.y0) / self.ny
        self.bin_area = self.bw * self.bh
        self.device = device

    @staticmethod
    def over_box(box, nx, ny, device):
        b = [float(v) for v in box]
        return Grid(b[0], b[1], b[2], b[3], nx, ny, device)

    def _deltas(self, c, origin, b, n):
        """For coordinates c: bin index k, and the two non-zeros of the step sequence g(c) at k and k + 1."""
        t = ((c - origin) / b).clamp(0, n)
        k = t.floor().clamp(max=n - 1)
        frac = (t - k) * b                       # part of bin k left of c
        first = b - frac                         # g(c) in bin k
        k = k.long()
        beyond = t >= n                          # c at/after the grid end: g = 0 everywhere
        first = torch.where(beyond, torch.zeros_like(first), first)
        second = torch.where(beyond, torch.zeros_like(frac), frac)
        return k, first, second

    def _corners(self, x0, y0, x1, y1):
        kx0, ax0, bx0 = self._deltas(x0, self.x0, self.bw, self.nx)
        kx1, ax1, bx1 = self._deltas(x1, self.x0, self.bw, self.nx)
        ky0, ay0, by0 = self._deltas(y0, self.y0, self.bh, self.ny)
        ky1, ay1, by1 = self._deltas(y1, self.y0, self.bh, self.ny)
        X = [(kx0, ax0, bx0, 1.0), (kx1, ax1, bx1, -1.0)]
        Y = [(ky0, ay0, by0, 1.0), (ky1, ay1, by1, -1.0)]
        return X, Y

    def corners(self, x0, y0, x1, y1):
        """The 16 (bin index, weight) entries of each rectangle, vectorized: idx (N, 16) into the (nx+1)(ny+1)
        array, w (N, 16) including the corner signs. Shared by accumulate_c and integrate_c."""
        kx0, ax0, bx0 = self._deltas(x0, self.x0, self.bw, self.nx)
        kx1, ax1, bx1 = self._deltas(x1, self.x0, self.bw, self.nx)
        ky0, ay0, by0 = self._deltas(y0, self.y0, self.bh, self.ny)
        ky1, ay1, by1 = self._deltas(y1, self.y0, self.bh, self.ny)
        # x: (N, 4) entries = corner x0 at k, k+1 (+), corner x1 at k, k+1 (-)
        kx = torch.stack([kx0, kx0 + 1, kx1, kx1 + 1], 1)
        wx = torch.stack([ax0, bx0, -ax1, -bx1], 1)
        ky = torch.stack([ky0, ky0 + 1, ky1, ky1 + 1], 1)
        wy = torch.stack([ay0, by0, -ay1, -by1], 1)
        idx = (kx[:, :, None] * (self.ny + 1) + ky[:, None, :]).reshape(-1, 16)
        w = (wx[:, :, None] * wy[:, None, :]).reshape(-1, 16)
        return idx, w

    def accumulate_c(self, c, value):
        idx, w = c
        D = torch.zeros((self.nx + 1) * (self.ny + 1), device=self.device, dtype=w.dtype)
        D.index_add_(0, idx.reshape(-1), (w * value[:, None]).reshape(-1))
        return D.view(self.nx + 1, self.ny + 1).cumsum(0).cumsum(1)[:self.nx, :self.ny]

    def integrate_c(self, fields, c):
        """fields (K, nx, ny) -> (N, K) integrals of each field over each rectangle."""
        idx, w = c
        K = fields.shape[0]
        R = torch.zeros(K, self.nx + 1, self.ny + 1, device=self.device, dtype=fields.dtype)
        R[:, :self.nx, :self.ny] = fields.flip(1).flip(2).cumsum(1).cumsum(2).flip(1).flip(2)
        g = R.view(K, -1)[:, idx]                       # (K, N, 16)
        return (g * w[None]).sum(2).T

    def accumulate(self, x0, y0, x1, y1, value, out=None):
        """map[i, j] = sum_r value_r * overlap_area(rect_r, bin_ij). value: per unit area (e.g. 1 for cell area)."""
        W = self.ny + 1
        D = torch.zeros((self.nx + 1) * W, device=self.device, dtype=value.dtype)
        X, Y = self._corners(x0, y0, x1, y1)
        idx, val = [], []
        for kx, ax, bx, sx in X:
            for ky, ay, by, sy in Y:
                s = sx * sy * value
                for dx, wx in ((0, ax), (1, bx)):
                    for dy, wy in ((0, ay), (1, by)):
                        idx.append((kx + dx) * W + (ky + dy)); val.append(s * wx * wy)
        D.index_add_(0, torch.cat(idx), torch.cat(val))
        m = D.view(self.nx + 1, W).cumsum(0).cumsum(1)[:self.nx, :self.ny]
        return m if out is None else out.add_(m)

    def integrate(self, field, x0, y0, x1, y1):
        """sum_ij overlap_area(rect_r, bin_ij) * field[i, j] for every rectangle r."""
        R = torch.zeros(self.nx + 1, self.ny + 1, device=self.device, dtype=field.dtype)
        R[:self.nx, :self.ny] = field.flip(0).flip(1).cumsum(0).cumsum(1).flip(0).flip(1)   # suffix sums
        Rf = R.view(-1); W = self.ny + 1
        X, Y = self._corners(x0, y0, x1, y1)
        tot = torch.zeros_like(x0)
        for kx, ax, bx, sx in X:
            for ky, ay, by, sy in Y:
                for dx, wx in ((0, ax), (1, bx)):
                    for dy, wy in ((0, ay), (1, by)):
                        tot = tot + (sx * sy) * wx * wy * Rf[(kx + dx) * W + (ky + dy)]
        return tot


def default_bins(n_objects, lo=16, hi=1024):
    """Power-of-two bin count per side, ~ sqrt(n) (as in ePlace / DREAMPlace)."""
    import math
    n = 2 ** int(round(math.log2(max(1.0, math.sqrt(max(1, n_objects))))))
    return max(lo, min(hi, n))
