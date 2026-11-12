"""Macro packing with a B*-tree and simulated annealing, for block-level floorplans (no rows).

A B*-tree (Chang et al. 2000) holds one block per node: the left child of a node is packed right next to it
(x = x_parent + w_parent), the right child above it at the same x; y comes from the contour of what is packed already
(preorder: a node, its left subtree, its right subtree). Every tree packs to the lower-left with no overlaps.

Start: shelf packing (blocks tallest first, rows as left chains while they fit the core width; a new row is the right
child of the first block of the row below). Moves: swap the blocks of two nodes; delete a node (one with at most one
child) and insert it as a child of another node. Two phases, each up to `moves` moves:
  1. cost = the area squeezed out of the core (blocks past its right or top edge), until it is 0;
  2. if the blocks fit: cost = HPWL, only among packings that still fit.
The blocks never overlap; when phase 1 does not reach 0 the best packing sticks out of the core by as little as found.
Blocks keep their orientation. Units: grid units (Design.macro_grid steps), on the CPU (numpy).
"""
import math
import random
import time

import numpy as np


class BStar:
    """B*-tree over nodes 0..n-1; node v holds block blk[v]."""

    def __init__(self, n):
        self.n = n
        self.left = np.full(n, -1); self.right = np.full(n, -1); self.parent = np.full(n, -1)
        self.blk = np.arange(n); self.root = 0

    def copy(self):
        t = BStar.__new__(BStar)
        t.n, t.root = self.n, self.root
        t.left, t.right, t.parent, t.blk = self.left.copy(), self.right.copy(), self.parent.copy(), self.blk.copy()
        return t

    def pack(self, kx, ky):
        """lower-left x, y per block (grid units); the contour is a height map over x"""
        x = np.zeros(self.n, dtype=np.int64); y = np.zeros(self.n, dtype=np.int64)
        hm = np.zeros(int(kx.sum()) + 2, dtype=np.int64)
        stack = [(self.root, 0)]
        left, right, blk = self.left, self.right, self.blk
        while stack:
            v, xv = stack.pop()
            b = blk[v]
            w = kx[b]
            yv = hm[xv:xv + w].max()
            hm[xv:xv + w] = yv + ky[b]
            x[b], y[b] = xv, yv
            if right[v] >= 0:
                stack.append((right[v], xv))
            if left[v] >= 0:
                stack.append((left[v], xv + w))
        return x, y

    def set_child(self, p, side, c):
        (self.left if side == 0 else self.right)[p] = c
        if c >= 0:
            self.parent[c] = p

    def swap(self, rng):
        a, b = rng.sample(range(self.n), 2)
        self.blk[a], self.blk[b] = self.blk[b], self.blk[a]
        return True

    def move(self, rng):
        """delete a node with at most one child, insert it as the left or right child of another node (the child
        that was there becomes its child on the same side); False if the drawn node has two children"""
        a = rng.randrange(self.n)
        if self.left[a] >= 0 and self.right[a] >= 0:
            return False
        c = self.left[a] if self.left[a] >= 0 else self.right[a]
        p = self.parent[a]
        if p < 0:
            if c < 0:
                return False
            self.root = c; self.parent[c] = -1
        else:
            self.set_child(p, 0 if self.left[p] == a else 1, c)
        self.left[a] = self.right[a] = self.parent[a] = -1
        t = rng.randrange(self.n - 1)
        t = t if t < a else t + 1
        side = rng.randrange(2)
        old = (self.left if side == 0 else self.right)[t]
        self.set_child(t, side, a)
        self.set_child(a, side, old)
        return True


def shelf(kx, ky, width):
    """the start: rows of blocks, tallest first, each row a left chain while it fits `width`"""
    n = len(kx)
    t = BStar(n)
    order = sorted(range(n), key=lambda b: (-ky[b], -kx[b], b))
    t.root = first = prev = order[0]
    used = kx[first]
    for b in order[1:]:
        if used + kx[b] <= width:
            t.set_child(prev, 0, b); used += kx[b]
        else:
            t.set_child(first, 1, b); first = b; used = kx[b]
        prev = b
    return t


def squeezed_out(x, y, kx, ky, ubx, uby):
    """area of the blocks past the last start of each (grid units^2)"""
    return float((np.maximum(x - ubx, 0) * ky + np.maximum(y - uby, 0) * kx).sum())


def anneal(kx, ky, ubx, uby, width, hpwl=None, moves=200, seed=0, verbose=False):
    """Pack blocks kx, ky (grid units, int) with last starts ubx, uby. hpwl(x, y) -> float for phase 2 (None: phase 1
    only). moves: per block and phase. -> (x, y, info)."""
    t0 = time.time()
    n = len(kx)
    rng = random.Random(seed)
    area = float((kx * ky).sum())
    tree = shelf(kx, ky, width)
    M = max(1, int(moves * n))

    def neighbour(t):
        if n < 2:
            return None
        t2 = t.copy()
        ok = t2.swap(rng) if rng.random() < 0.5 else t2.move(rng)
        return t2 if ok else None

    def run(t, cost_of, feasible, budget, stop_at_zero):
        x, y = t.pack(kx, ky)
        cost = cost_of(x, y)
        best = (cost, t.copy())
        if (stop_at_zero and cost == 0) or n < 2:
            return best, 0
        ups = []                                          # start temperature: an uphill move accepted with p ~ 0.5
        for _ in range(30):
            t2 = neighbour(t)
            if t2 is None:
                continue
            x2, y2 = t2.pack(kx, ky)
            if feasible(x2, y2):
                dlt = cost_of(x2, y2) - cost
                if dlt > 0:
                    ups.append(dlt)
        T = (float(np.mean(ups)) if ups else 1e-3) / math.log(2)
        alpha = 1e-4 ** (1.0 / budget)
        s = 0
        for s in range(budget):
            T *= alpha
            t2 = neighbour(t)
            if t2 is None:
                continue
            x2, y2 = t2.pack(kx, ky)
            if not feasible(x2, y2):
                continue
            c2 = cost_of(x2, y2)
            if c2 <= cost or rng.random() < math.exp(-(c2 - cost) / T):
                t, cost = t2, c2
                if cost < best[0]:
                    best = (cost, t.copy())
                    if stop_at_zero and cost == 0:
                        break
        return best, s + 1

    # phase 1: into the core
    out = lambda x, y: squeezed_out(x, y, kx, ky, ubx, uby) / area
    (c1, tree), m1 = run(tree, out, lambda x, y: True, M, True)
    fits = c1 == 0
    m2 = 0
    h0 = h1 = None
    if fits and hpwl is not None:
        # phase 2: HPWL, among packings that fit
        x, y = tree.pack(kx, ky)
        h0 = hpwl(x, y)
        (c2, tree), m2 = run(tree, lambda x, y: hpwl(x, y) / h0, lambda x, y: squeezed_out(x, y, kx, ky, ubx, uby) == 0,
                             M, False)
        h1 = c2 * h0
    x, y = tree.pack(kx, ky)
    info = {'fits': fits, 'squeezed_out': c1, 'moves': (m1, m2), 'hpwl_fit': h0, 'hpwl': h1,
            'seconds': time.time() - t0}
    if verbose:
        print('    b*-tree: %s after %d moves%s, %.2fs' % (
            'fits' if fits else 'squeezed out %.2f%% of the block area' % (100 * c1), m1,
            ', HPWL %.4g -> %.4g in %d moves' % (h0, h1, m2) if h0 is not None else '', info['seconds']))
    return x, y, info
