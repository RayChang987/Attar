#!/usr/bin/env python3
"""Validate the distillation on routed pa_pdn runs.

1. HPWL on the GPU from the distilled final DEF vs Innovus' own 'Total net bbox length' before routeDesign.
2. Write-back of the unchanged placement: COMPONENTS identical to the source.
3. Write-back of a perturbed placement (shifts and flips): re-distilled HPWL equals the in-memory HPWL.
usage: validate_pa_pdn.py <scratch dir> [design ...]"""
import glob
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import attar  # noqa: E402

HOME = os.path.expanduser('~')
OUT = HOME + '/i3dPDK_NS3K/innovus_enablement/pnr/out'
ENA = HOME + '/i3dPDK_NS3K/innovus_enablement'
LEFS = [ENA + '/lef_x4/i3dPDK_NS3K_5T_tech_unidir_dir.lef', ENA + '/lef_x4/NS3K_5T_cells.lef'] + sorted(glob.glob(ENA + '/lef_x4/fakeram*.lef'))


def innovus_bbox(run):
    """Last 'Total net bbox length' before routeDesign (um, x4 geometry)."""
    last = None
    for line in open(run + '/innovus.log', errors='ignore'):
        if line.startswith('Total net bbox length'):
            last = float(line.split()[5])
        elif '<CMD> routeDesign' in line:
            break
    return last


def components(path):
    out = []; on = False
    for line in open(path, errors='replace'):
        if line.startswith('COMPONENTS'): on = True
        elif line.startswith('END COMPONENTS'): break
        elif on: out.append(line)
    return out


def sync(dev):
    if dev.type == 'cuda': torch.cuda.synchronize()


if __name__ == '__main__':
    tmp = sys.argv[1]
    designs = sys.argv[2:] or ['gcd', 'aes', 'jpeg', 'riscv32i', 'bp_fe_top', 'ariane133']
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device', dev)
    for d in designs:
        runs = [r for r in sorted(glob.glob('%s/%s/NS3K_5T/ns3k_pl_*_cNF_fast' % (OUT, d)))
                if [p for p in glob.glob(r + '/*.def') if 'floorplan' not in p]]
        if not runs:
            print(d, 'no routed NF run'); continue
        run = runs[0]
        src = [p for p in glob.glob(run + '/*.def') if 'floorplan' not in p][0]
        ir = os.path.join(tmp, d)
        t = time.time(); attar.distill(src, LEFS, ir); t_dist = time.time() - t
        x = attar.load(ir, dev)
        h = x.hpwl(); sync(dev)
        t = time.time()
        for _ in range(20): h = x.hpwl()
        sync(dev); t_hp = (time.time() - t) / 20
        inv = innovus_bbox(run)
        hv = h.item()
        print('\n== %s (%s): %d cells, %d nets | distill %.1fs, GPU HPWL %.2f ms' % (
            d, os.path.basename(run), len(x.pos), int(x.n_mask.sum()), t_dist, 1e3 * t_hp))
        print('  HPWL attar %.1f um | innovus bbox %.1f um | diff %+.3f%%' % (hv, inv, 100 * (hv - inv) / inv) if inv else '  HPWL attar %.1f' % hv)

        # 2. unchanged write-back
        out = os.path.join(tmp, d + '.same.def')
        n = x.export(out)
        a, b = components(src), components(out)
        diff = sum(1 for u, v in zip(a, b) if u != v) + abs(len(a) - len(b))
        print('  write-back unchanged: %d lines needed rewriting, %d COMPONENTS lines differ; other sections identical: %s' % (
            n, diff, open(src, errors='replace').read().replace(''.join(a), '') == open(out, errors='replace').read().replace(''.join(b), '')))

        # 3. perturbed write-back: shift 10% of movable cells by +-3 sites and change orientations, then re-read
        g = torch.Generator(device='cpu').manual_seed(0)
        mv = torch.nonzero(x.movable).flatten().cpu()
        pick = mv[torch.randperm(len(mv), generator=g)[:max(1, len(mv) // 10)]].to(dev)
        site = 0.176  # um (x4 site width 352 dbu)
        x.pos[pick, 0] += (torch.randint(-3, 4, (len(pick),), generator=g).to(dev).float() * site)
        x.orient[pick] = torch.tensor([0, 1, 4, 5], device=dev)[torch.randint(0, 4, (len(pick),), generator=g).to(dev)]
        x.update_orient()
        h1 = x.hpwl().item()
        out = os.path.join(tmp, d + '.moved.def')
        x.export(out)
        attar.distill(out, LEFS, ir + '_moved')
        h2 = attar.load(ir + '_moved', dev).hpwl().item()
        print('  write-back perturbed (%d cells): in-memory %.3f um, re-distilled %.3f um, diff %.2e' % (len(pick), h1, h2, abs(h1 - h2) / h1))
