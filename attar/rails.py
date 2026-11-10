"""Power rail parity (after OpenROAD dpl: network.cxx, dbToOpendp.cpp inferR0RowPower, Architecture::powerCompatible).

  master rails   bottom / top rail of each master from its POWER / GROUND pin shapes: the lower-most of the two
                 is the bottom rail, the upper-most the top rail (UNK without both)
  row rails      an R0 (N) row has the rails of a single-row core master whose rails differ (the reference);
                 rows whose orientation flips Y (FS, S) swap them
  compatible     a cell starting in row r spans rows r..r+k-1; it fits if (bottom, top) == (row r bottom, row
                 r+k-1 top), or after a flip (swapped); unknown rails always fit. Single-row cells always fit
                 (they flip to the row).
Codes: 0 unknown, 1 VSS (ground), 2 VDD (power).
"""
import torch

from .distill import U_GROUND, U_POWER

UNK, VSS, VDD = 0, 1, 2
FLIPS_Y = (1, 5)            # S (R180), FS (MX) in deffile.ORIENTS


def master_rails(d):
    """(bottom, top) rail code of every master (long tensors on d.device), from the PG pin boxes in the IR."""
    a = d.raw
    M = len(a['m_size'])
    dev = d.device
    if 'm_pin_use' not in a or 'pin_offsets' not in d.facts:
        z = torch.zeros(M, dtype=torch.long, device=dev)
        return z, z.clone()
    ptr = torch.as_tensor(a['m_pin_ptr'].astype('int64'), device=dev)
    use = torch.as_tensor(a['m_pin_use'].astype('int64'), device=dev)
    box = torch.as_tensor(a['m_pin_box'].astype('int64'), device=dev)
    owner = torch.repeat_interleave(torch.arange(M, device=dev), ptr[1:] - ptr[:-1])
    big = 2 ** 40
    out = {}
    for kind, code in (('pwr', U_POWER), ('gnd', U_GROUND)):
        m = use == code
        lo = torch.full((M,), big, dtype=torch.long, device=dev).scatter_reduce(0, owner[m], box[m, 1], 'amin')
        hi = torch.full((M,), -big, dtype=torch.long, device=dev).scatter_reduce(0, owner[m], box[m, 3], 'amax')
        out[kind] = (lo, hi)
    (plo, phi), (glo, ghi) = out['pwr'], out['gnd']
    known = (plo < big) & (glo < big)
    bot = torch.where(plo < glo, torch.full_like(plo, VDD), torch.full_like(plo, VSS))
    top = torch.where(phi > ghi, torch.full_like(plo, VDD), torch.full_like(plo, VSS))
    zero = torch.zeros_like(bot)
    return torch.where(known, bot, zero), torch.where(known, top, zero)


def reference_r0(d, mbot, mtop, row_h):
    """(bottom, top) of an R0 row: the most used single-row core master whose rails differ, else (UNK, UNK)."""
    a = d.raw
    dev = d.device
    m_h = torch.as_tensor(a['m_size'][:, 1].astype('float64'), device=dev) * d.scale
    single = ((m_h - row_h).abs() < 1e-6) & (mbot != UNK) & (mbot != mtop)
    single = single & torch.as_tensor(a['m_class'] == 0, device=dev)
    if not bool(single.any()):
        return UNK, UNK
    cm = torch.as_tensor(a['c_master'].astype('int64'), device=dev)
    use = torch.bincount(cm[cm >= 0], minlength=len(m_h)).to(dev)
    i = int(torch.where(single, use, torch.full_like(use, -1)).argmax())
    return int(mbot[i]), int(mtop[i])


def row_rails(ref_bot, ref_top, row_orient):
    """(bottom, top) of every row from its orientation."""
    flip = torch.zeros_like(row_orient, dtype=torch.bool)
    for o in FLIPS_Y:
        flip |= row_orient == o
    b = torch.where(flip, torch.full_like(row_orient, ref_top), torch.full_like(row_orient, ref_bot))
    t = torch.where(flip, torch.full_like(row_orient, ref_bot), torch.full_like(row_orient, ref_top))
    return b, t


def _eq(a, b):
    return (a == b) | (a == UNK) | (b == UNK)


def compatible(cbot, ctop, k, rbot, rtop):
    """For cells with rails (cbot, ctop) and height k rows: (fits as is, fits flipped) at every start row.
    Returns two (n, R) boolean tensors; rows where r + k > R are False. Single-row cells always fit (flip)."""
    R = len(rbot)
    r = torch.arange(R, device=rbot.device)
    top_row = (r[None, :] + k[:, None] - 1).clamp(max=R - 1)
    inside = (r[None, :] + k[:, None]) <= R
    rb = rbot[None, :].expand(len(k), -1)
    rt = rtop[top_row]
    keep = _eq(cbot[:, None], rb) & _eq(ctop[:, None], rt) & inside
    flip = _eq(ctop[:, None], rb) & _eq(cbot[:, None], rt) & inside
    single = (k == 1)[:, None]
    return keep | (single & inside & ~flip), flip | (single & inside & ~keep)


def rows_spanned(ys, hb, y0, y1, tol=None):
    """[rlo, rhi) of the base rows an object [y0, y1) overlaps: ys[r] + hb > y0 and ys[r] < y1, with a small
    tolerance so float32 edges on the row grid neither add nor drop a row."""
    t = 1e-3 * hb if tol is None else tol
    rlo = torch.searchsorted(ys, y0 + t - hb, right=True)
    rhi = torch.searchsorted(ys, y1 - t, right=False)
    return rlo, rhi


def base_rows(d):
    """Base rows (smallest row height), one per distinct y: (row height, ys, x0, x1, site width, orientation)."""
    hb = d.row_h.min()
    base = d.row_h <= hb + 1e-6
    ys, inv = torch.unique(d.row_xy[base, 1], return_inverse=True)
    dev = d.device
    x0 = torch.full((len(ys),), float('inf'), device=dev).scatter_reduce(0, inv, d.row_box[base, 0], 'amin')
    x1 = torch.full((len(ys),), -float('inf'), device=dev).scatter_reduce(0, inv, d.row_box[base, 2], 'amax')
    sw = d.row_site_w[base]
    if (sw - sw[0]).abs().max() > 1e-6:
        raise NotImplementedError('rows with different site widths')
    orient = torch.zeros(len(ys), dtype=torch.long, device=dev)
    if d.row_orient is not None:
        orient.scatter_(0, inv, d.row_orient[torch.nonzero(base).flatten()])
    return hb.item(), ys, x0, x1, sw[0].item(), orient


def info(d, base_rows=None):
    """Everything the legalizer and the checker need: per-cell (bottom, top), per-base-row (bottom, top), and whether
    rails are known at all. base_rows: (ys, orient) of the base rows; default from the design's rows."""
    mbot, mtop = master_rails(d)
    hb = d.row_h.min().item()
    rb, rt = reference_r0(d, mbot, mtop, hb)
    cm = torch.as_tensor(d.raw['c_master'].astype('int64'), device=d.device)
    cbot = torch.where(cm >= 0, mbot[cm.clamp(min=0)], torch.zeros_like(cm))
    ctop = torch.where(cm >= 0, mtop[cm.clamp(min=0)], torch.zeros_like(cm))
    if base_rows is None:
        _, ys, _, _, _, orient = globals()['base_rows'](d)
    else:
        ys, orient = base_rows
    rbot, rtop = row_rails(rb, rt, orient)
    return {'known': rb != UNK, 'cell_bot': cbot, 'cell_top': ctop, 'row_bot': rbot, 'row_top': rtop, 'ref': (rb, rt)}


def row_gaps(d, ys, tol=1e-3):
    """Gaps between the row segments of each base row: (row index into ys, x0, x1) in microns. A base row is covered
    by its own segments and by every taller row (multi-height row sets) that spans it. Rows cut around macros
    (ORFS / OpenROAD cut them at the macro halo) leave gaps; a cell in one is outside the rows (OpenROAD
    check_placement DPL-0004)."""
    dev = ys.device
    hb = d.row_h.min()
    R = len(ys)
    lo = torch.searchsorted(ys, d.row_xy[:, 1] - tol).clamp(max=R)
    k = (d.row_h / hb).round().long().clamp(min=1)
    ok = (lo < R)
    ok &= (ys[lo.clamp(max=R - 1)] - d.row_xy[:, 1]).abs() < tol             # rows on the base-row grid
    lo, k = lo[ok], k[ok]
    seg = torch.nonzero(ok).flatten()
    rep = torch.repeat_interleave(torch.arange(len(seg), device=dev), k)
    r = lo[rep] + (torch.arange(len(rep), device=dev) - torch.repeat_interleave(torch.cumsum(k, 0) - k, k))
    keep = r < R
    r, x0, x1 = r[keep], d.row_box[seg[rep[keep]], 0], d.row_box[seg[rep[keep]], 2]
    big = float((x1.max() - x0.min()).abs().item()) * 4 + 1
    o = torch.argsort(r.double() * big + (x0 - x0.min()).double())
    r, x0, x1 = r[o], x0[o], x1[o]
    run = torch.cummax(r.double() * big + x1.double(), 0).values - r.double() * big       # covered so far, per row
    same = torch.cat([torch.zeros(1, dtype=torch.bool, device=dev), r[1:] == r[:-1]])
    prev_end = torch.cat([torch.zeros(1, dtype=run.dtype, device=dev), run[:-1]])
    gap = same & (x0.double() > prev_end + tol)
    return r[gap], prev_end[gap].float(), x0[gap]
