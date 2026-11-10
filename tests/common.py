"""Shared helpers for the Attar tests. Run from the repo root: python3 -m unittest discover -s tests -v

The tests need the local zoo (zoo/orfs_asap7, zoo/ispd2005, zoo/ns3k) and the files it points to; a missing design
skips its tests."""
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import attar  # noqa: E402

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
SMALL, MEDIUM = 'orfs_asap7/gcd', 'orfs_asap7/aes'


def need(name):
    try:
        return attar.load(name, DEV)
    except FileNotFoundError as e:
        raise unittest.SkipTest(str(e))
MACRO_DESIGNS = ['orfs_asap7/riscv32i', 'orfs_asap7/tinyRocket', 'orfs_asap7/swerv_wrapper', 'orfs_asap7/cva6']
RUNS = os.path.join(os.path.dirname(__file__), '..', 'orfs_runs')

def gpl_final_hpwl(name):
    d = name.split('/')[1]
    log = os.path.join(RUNS, d + '_base', 'logs', 'asap7', d, 'base', '3_3_place_gp.log')
    if not os.path.exists(log):
        return None
    with open(log) as f:
        for line in f:
            if 'Final HPWL' in line:
                return float(line.split()[-1])
BOOKSHELF = 'ispd2005/adaptec1'

def bookshelf_hpwl_numpy(aux, pl):
    """Independent HPWL straight from the Bookshelf files (pin offsets from the node center), no Attar code."""
    d = os.path.dirname(aux); base = os.path.splitext(os.path.basename(aux))[0]
    size = {}
    for line in open(os.path.join(d, base + '.nodes')):
        t = line.split()
        if len(t) >= 3 and not line.startswith(('UCLA', '#', 'Num')):
            size[t[0]] = (float(t[1]), float(t[2]))
    loc = {}
    for line in open(pl):
        t = line.split()
        if len(t) >= 3 and not line.startswith(('UCLA', '#')):
            loc[t[0]] = (float(t[1]), float(t[2]))
    total = 0.0; xs = []; ys = []
    def flush():
        nonlocal total
        if len(xs) > 1:
            total += max(xs) - min(xs) + max(ys) - min(ys)
    for line in open(os.path.join(d, base + '.nets')):
        if line.startswith('NetDegree'):
            flush(); xs.clear(); ys.clear(); continue
        t = line.replace(':', ' ').split()
        if not t or t[0] not in size:
            continue
        w, h = size[t[0]]; x, y = loc[t[0]]
        ox, oy = (float(t[2]), float(t[3])) if len(t) >= 4 else (0.0, 0.0)
        xs.append(x + w / 2 + ox); ys.append(y + h / 2 + oy)
    flush()
    return total
