#!/usr/bin/env python3
"""README diagrams as SVG (docs/images): flow.svg (how Attar works) and roadmap.svg (the roadmap as a mind map).
Warm light palette, the same as attar.viz: paper surface, terracotta for Attar, sand for data, amber accents.
Arrow labels sit on their arrows as pills in the arrow's color; arrowheads take their line's color and stop short
of the box they point at.
usage: make_readme_diagrams.py"""
import os

IMG = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'docs', 'images')
C = dict(paper='#fbf8f3', card='#fffdf9', ink='#2b211c', ink2='#6b5d52', muted='#a39486', line='#b9a993',
         terra='#d9622b', terra_d='#b44a1c', terra_l='#f9e6d8', sand='#ecdfc8', sand_d='#a8946f', slow='#f1e9dd',
         amber='#d99a1e', white='#fffaf4')
SANS = "font-family=\"'Segoe UI', Helvetica, Arial, sans-serif\""
MONO = "font-family=\"'SFMono-Regular', Consolas, 'DejaVu Sans Mono', monospace\""
ARROWS = {'ink': C['ink2'], 'terra': C['terra'], 'amber': C['amber']}


def esc(t):
    return t.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def text(x, y, t, size=16, fill=C['ink'], weight='normal', anchor='middle', mono=False):
    return ('<text x="%g" y="%g" font-size="%g" fill="%s" font-weight="%s" text-anchor="%s" %s>%s</text>'
            % (x, y, size, fill, weight, anchor, MONO if mono else SANS, esc(t)))


def box(x, y, w, h, title, lines=(), fill=C['card'], stroke=C['line'], tcol=C['ink'], lcol=C['ink2'], r=14,
        tsize=24, lsize=17, mono_title=False, mono_lines=False, dash=None, shadow=True):
    s = []
    if shadow:
        s.append('<rect x="%g" y="%g" width="%g" height="%g" rx="%g" fill="#000" opacity="0.045"/>' % (x + 2, y + 4, w, h, r))
    s.append('<rect x="%g" y="%g" width="%g" height="%g" rx="%g" fill="%s" stroke="%s" stroke-width="1.6"%s/>'
             % (x, y, w, h, r, fill, stroke, ' stroke-dasharray="%s"' % dash if dash else ''))
    n = len(lines)
    top = y + h / 2 - (tsize + n * (lsize + 7)) / 2 + tsize * 0.85
    s.append(text(x + w / 2, top, title, tsize, tcol, 'bold', mono=mono_title))
    for i, l in enumerate(lines):
        s.append(text(x + w / 2, top + 10 + (i + 1) * (lsize + 6), l, lsize, lcol, mono=mono_lines))
    return '\n'.join(s)


def pill(cx, cy, label, color, size=16):
    w = len(label) * size * 0.6 + 26
    return ('<rect x="%g" y="%g" width="%g" height="%g" rx="%g" fill="%s"/>' % (cx - w / 2, cy - 15, w, 30, 15, color)
            + text(cx, cy + size * 0.36, label, size, C['white'], 'bold'))


def arrow(points, kind='ink', dash=None, label=None, at=None):
    """A polyline ending in an arrowhead of its own color; label: a pill centered at `at` (a point on the line)."""
    color = ARROWS[kind]
    d = 'M ' + ' L '.join('%g %g' % p for p in points)
    s = ['<path d="%s" fill="none" stroke="%s" stroke-width="2.4"%s stroke-linejoin="round" marker-end="url(#ah_%s)"/>'
         % (d, color, ' stroke-dasharray="%s"' % dash if dash else '', kind)]
    if label:
        s.append(pill(at[0], at[1], label, color))
    return '\n'.join(s)


def defs():
    m = ''.join('<marker id="ah_%s" viewBox="0 0 12 12" refX="11" refY="6" markerWidth="15" markerHeight="15" '
                'markerUnits="userSpaceOnUse" orient="auto"><path d="M 0 0 L 12 6 L 0 12 z" fill="%s"/></marker>'
                % (k, c) for k, c in ARROWS.items())
    return ('<defs>' + m + '<filter id="glow" x="-50%" y="-50%" width="200%" height="200%">'
            '<feGaussianBlur stdDeviation="7"/></filter></defs>')


def flow():
    W, H = 1280, 870
    s = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="%d" height="%d">' % (W, H, W, H), defs(),
         '<rect width="%d" height="%d" rx="22" fill="%s"/>' % (W, H, C['paper'])]
    # regions: the fast loop (Attar) and the slow loop (ORFS)
    s.append('<rect x="770" y="28" width="482" height="814" rx="22" fill="%s"/>' % C['terra_l'])
    s.append(text(796, 66, 'Fast loop', 19, C['terra_d'], 'bold', 'start'))
    s.append(text(796 + 9 * 19 * 0.62 + 16, 66, 'Seconds, on the GPU', 17, C['terra_d'], anchor='start'))
    s.append('<rect x="28" y="494" width="712" height="348" rx="22" fill="%s"/>' % C['slow'])
    s.append(text(54, 532, 'Slow loop', 19, C['sand_d'], 'bold', 'start'))
    s.append(text(54 + 9 * 19 * 0.62 + 16, 532, 'Minutes to hours, ORFS + ASAP7', 17, C['sand_d'], anchor='start'))
    # sources -> IR
    s.append(text(54, 66, 'Any source', 19, C['ink2'], 'bold', 'start'))
    src = [('LEF / DEF', 'ORFS, ISPD 15/18/19/24, ICCAD 25'), ('Bookshelf', 'ISPD 05, IBM-HB+, GSRC'),
           ('YAL, GSRC blocks', 'MCNC, GSRC')]
    for i, (t, l) in enumerate(src):
        y = 92 + i * 116
        s.append(box(40, y, 286, 96, t, [l], stroke=C['sand_d'], tsize=22, lsize=15.5))
        s.append('<path d="M 326 %d L 350 %d" stroke="%s" stroke-width="2.4"/>' % (y + 48, y + 48, C['ink2']))
    s.append('<path d="M 350 140 L 350 372" stroke="%s" stroke-width="2.4"/>' % C['ink2'])
    s.append(box(490, 170, 210, 172, 'Attar IR', ['Safetensors', '+ manifest', 'Facts +', 'provenance'], fill=C['sand'],
                 stroke=C['sand_d']))
    s.append(arrow([(350, 256), (486, 256)], label='Distill', at=(414, 256)))
    # Design (the hub), engines, evaluator, your algorithm
    s.append(box(810, 104, 410, 140, 'Design', ['GPU tensors: cells, pins, nets, rows', 'Facts → what it can do'],
                 fill=C['terra'], stroke=C['terra_d'], tcol=C['white'], lcol='#fde7d6', tsize=30, lsize=17.5))
    s.append(arrow([(700, 198), (806, 198)], label='Load', at=(744, 198)))
    s.append(box(792, 360, 222, 206, 'attar.naive', ['make_die', 'place_io', 'place_macros', 'global_place', 'legalize'],
                 mono_title=True, mono_lines=True, stroke=C['terra'], tsize=22, lsize=16))
    s.append(box(1030, 360, 210, 206, 'Evaluator', ['HPWL', 'Steiner', 'RUDY', 'Overflow', 'Legality'],
                 stroke=C['terra'], lsize=16))
    s.append(box(792, 620, 222, 104, 'Your algorithm', ['A function of', 'the tensors'], dash='7 6', stroke=C['terra'],
                 tsize=21, lsize=16, shadow=False))
    s.append('<path d="M 903 620 L 903 566" stroke="%s" stroke-width="2" stroke-dasharray="6 5"/>' % C['terra'])
    s.append(arrow([(850, 244), (850, 354)], 'terra', label='Move', at=(850, 300)))
    s.append(arrow([(966, 360), (966, 250)], 'terra', label='New facts', at=(966, 300)))
    s.append(arrow([(1135, 244), (1135, 354)], 'terra', label='Measure', at=(1135, 300)))
    # export -> DEF -> ORFS -> truth
    s.append(arrow([(810, 232), (748, 232), (748, 452), (130, 452), (130, 600)], label='Export', at=(440, 452)))
    s.append(box(44, 606, 172, 136, 'DEF', ['Placement', 'Floorplan']))
    s.append(box(282, 576, 204, 196, 'ORFS', ['Tapcell, PDN', 'CTS, routing', 'STA'], stroke=C['sand_d'], tsize=25))
    s.append(box(552, 576, 168, 196, 'Truth', ['Routed WL', 'DRC', 'WNS, TNS'], fill=C['sand'], stroke=C['sand_d']))
    s.append(arrow([(216, 674), (276, 674)]))
    s.append(arrow([(486, 674), (546, 674)]))
    # agreement
    s.append(arrow([(1135, 566), (1135, 808), (636, 808), (636, 778)], 'amber', dash='9 7', label='Do they agree?',
                   at=(900, 808)))
    s.append('</svg>')
    open(os.path.join(IMG, 'flow.svg'), 'w').write('\n'.join(s))


def roadmap():
    """Mind map in the logo's palette (dark roast ground; yellow -> orange -> coral; gold, cream).
    Attar in the middle; Data to the left (placement and floorplan benchmarks in two colors, with design counts),
    Naive Engine and Evaluation to the right. Leaves carry a status chip: Done, Not yet (dashed), Won't have (dim)."""
    K = dict(bg0='#34201a', bg1='#1d110c', text='#f6e7cf', muted='#9c8576', dim='#6f5a4d', yellow='#ffc95a',
             orange='#ff9a3c', coral='#f26446', gold='#d9b05a', cream='#ffe3a8', brick='#ec6a4f', ink='#24150f')
    F = "font-family=\"'Segoe UI', 'Helvetica Neue', Helvetica, Arial, sans-serif\""

    def t(x, y, s, size, fill, weight='normal', anchor='middle', ls=None):
        return ('<text x="%g" y="%g" font-size="%g" fill="%s" font-weight="%s" text-anchor="%s"%s %s>%s</text>'
                % (x, y, size, fill, weight, anchor, ' letter-spacing="%g"' % ls if ls else '', F, esc(s)))

    PL, FP = K['cream'], K['brick']                       # placement / floorplan benchmarks
    DATA, NAIVE, EVAL = K['cream'], K['orange'], K['gold']
    data = [('Contest benchmarks', [('ISPD 2005', 8, PL), ('ISPD 2015', 20, PL), ('ISPD 2018', 10, PL),
                                    ('ISPD 2019', 10, PL), ('ISPD 2024', 5, PL), ('ICCAD 2025', 13, PL),
                                    ('GSRC placement', 4, PL), ('MCNC', 10, FP), ('GSRC blocks', 12, FP),
                                    ('IBM-HB+ (ISPD 2006)', 28, FP)]),
            ('ASAP7', [(15, PL), (5, FP)]),                  # stops here: design counts only
            ('NS3K', [(99, PL)])]
    naive = [('Floorplan', [('Packing', 'Done'), ('Macro placement', 'Done'), ('IO placement', 'Done')]),
             ('Placement', [('Placer', 'Done'), ('Legalizer', 'Done'), ('PDN-aware refinement', 'Done')])]
    ev = [('HPWL', 'Done'), ('OVF', 'Done'), ('Timing', 'Done'), ('Blocked pin checking', 'Not yet'),
          ('Detailed routing', "Won't have"), ('DRV checks', "Won't have")]

    FL, FM, FC = 23, 23, 27                               # font: leaf, middle, category
    PITCH, LH, GAP = 58, 46, 30
    LW_D, LW, LW_E = 368, 384, 420
    MW_D, CW, RX, RY, CW_R, MW_R = 290, 192, 98, 52, 236, 168
    x_dleaf = 40
    x_dmid = x_dleaf + LW_D + 38
    x_dcat = x_dmid + MW_D + 36
    CX = x_dcat + CW + 36 + RX
    x_rcat = CX + RX + 40
    x_rmid = x_rcat + CW_R + 38
    x_nleaf = x_rmid + MW_R + 38
    x_eleaf = x_rmid
    W = x_nleaf + LW + 50
    top = 50
    h_left = (len(data[0][1]) + len(data) - 2) * PITCH + GAP
    h_right = 11 * PITCH + 2 * GAP
    Ht = max(h_left, h_right)
    cy = top + LH / 2 + Ht / 2
    H = int(top + LH + Ht + 160)

    s = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="%d" height="%d">' % (W, H, W, H),
         '<defs>'
         '<radialGradient id="bg" cx="0.5" cy="0.42" r="0.75"><stop offset="0" stop-color="%s"/>'
         '<stop offset="1" stop-color="%s"/></radialGradient>'
         '<pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse"><path d="M 40 0 L 0 0 0 40" '
         'fill="none" stroke="#ffffff" stroke-opacity="0.028" stroke-width="1"/></pattern>'
         '<linearGradient id="brand" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="%s"/>'
         '<stop offset="0.55" stop-color="%s"/><stop offset="1" stop-color="%s"/></linearGradient>'
         '<radialGradient id="glow" r="0.5"><stop offset="0.35" stop-color="%s" stop-opacity="0.35"/>'
         '<stop offset="1" stop-color="%s" stop-opacity="0"/></radialGradient>'
         '<filter id="soft" x="-20%%" y="-40%%" width="140%%" height="180%%"><feGaussianBlur in="SourceAlpha" stdDeviation="5"/>'
         '<feOffset dy="4"/><feComponentTransfer><feFuncA type="linear" slope="0.45"/></feComponentTransfer>'
         '<feMerge><feMergeNode/><feMergeNode in="SourceGraphic"/></feMerge></filter>'
         '</defs>' % (K['bg0'], K['bg1'], K['yellow'], K['orange'], K['coral'], K['orange'], K['orange']),
         '<rect width="%d" height="%d" rx="28" fill="url(#bg)"/>' % (W, H),
         '<rect width="%d" height="%d" rx="28" fill="url(#grid)"/>' % (W, H)]
    links, nodes = [], []

    def link(x1, y1, x2, y2, col, wd, op=0.6):
        mx = (x1 + x2) / 2
        links.append('<path d="M %g %g C %g %g %g %g %g %g" fill="none" stroke="%s" stroke-width="%g" '
                     'stroke-linecap="round" stroke-opacity="%g"/>' % (x1, y1, mx, y1, mx, y2, x2, y2, col, wd, op))

    def solid(x, y, w, h, label, col, size):              # category: filled in its branch color; '\n' breaks lines
        lines = label.split('\n')
        nodes.append('<rect x="%g" y="%g" width="%g" height="%g" rx="%g" fill="%s" filter="url(#soft)"/>'
                     % (x, y - h / 2, w, h, min(h / 2, 30), col))
        for i, l in enumerate(lines):
            dy = (i - (len(lines) - 1) / 2) * size * 1.15
            nodes.append(t(x + w / 2, y + dy + size * 0.36, l, size, K['ink'], 'bold'))

    def ring(x, y, w, h, label, col, size):               # middle level: outlined
        nodes.append('<rect x="%g" y="%g" width="%g" height="%g" rx="%g" fill="%s" fill-opacity="0.10" stroke="%s" '
                     'stroke-width="2"/>' % (x, y - h / 2, w, h, h / 2, col, col))
        nodes.append(t(x + w / 2, y + size * 0.36, label, size, col, 'bold'))

    def chip(x, y, label, kind, col):
        w = len(label) * 9.6 + 26
        if kind == 'Done':
            nodes.append('<rect x="%g" y="%g" width="%g" height="28" rx="14" fill="%s"/>' % (x, y - 14, w, col))
            nodes.append(t(x + w / 2, y + 5.8, label, 16, K['ink'], 'bold'))
        elif kind == 'Not yet':
            nodes.append('<rect x="%g" y="%g" width="%g" height="28" rx="14" fill="none" stroke="%s" stroke-width="1.6" '
                         'stroke-dasharray="4 3"/>' % (x, y - 14, w, col))
            nodes.append(t(x + w / 2, y + 5.8, label, 16, col, 'bold'))
        else:
            nodes.append('<rect x="%g" y="%g" width="%g" height="28" rx="14" fill="#ffffff" fill-opacity="0.07"/>'
                         % (x, y - 14, w))
            nodes.append(t(x + w / 2, y + 5.8, label, 16, K['muted'], 'bold'))
        return w

    def leaf(x, y, w, label, col, status=None, counts=()):
        if status == "Won't have":
            nodes.append('<rect x="%g" y="%g" width="%g" height="%g" rx="%g" fill="#ffffff" fill-opacity="0.035" '
                         'stroke="%s" stroke-opacity="0.5" stroke-width="1.4"/>' % (x, y - LH / 2, w, LH, LH / 2, K['dim']))
            tc = K['muted']
        elif status == 'Not yet':
            nodes.append('<rect x="%g" y="%g" width="%g" height="%g" rx="%g" fill="none" stroke="%s" stroke-width="1.8" '
                         'stroke-dasharray="6 5"/>' % (x, y - LH / 2, w, LH, LH / 2, col))
            tc = K['text']
        else:
            nodes.append('<rect x="%g" y="%g" width="%g" height="%g" rx="%g" fill="%s" fill-opacity="0.13" stroke="%s" '
                         'stroke-opacity="0.85" stroke-width="1.6"/>' % (x, y - LH / 2, w, LH, LH / 2, col, col))
            tc = K['text']
        nodes.append(t(x + 22, y + FL * 0.36, label, FL, tc, anchor='start'))
        if status:
            cw = len(status) * 9.6 + 26
            chip(x + w - cw - 9, y, status, status, col)
        cx = x + w - 28
        for n, c in reversed(counts):
            nodes.append('<circle cx="%g" cy="%g" r="17" fill="%s"/>' % (cx, y, c))
            nodes.append(t(cx, y + 5.8, str(n), 16, K['ink'], 'bold'))
            cx -= 40

    # ---- left: Data
    y = cy - h_left / 2
    gname, kids = data[0]                                 # Contest benchmarks: one leaf per suite
    ys = [y + i * PITCH for i in range(len(kids))]
    my = (ys[0] + ys[-1]) / 2
    for yy, (label, n, col) in zip(ys, kids):
        link(x_dmid, my, x_dleaf + LW_D, yy, col, 2.2)
        leaf(x_dleaf, yy, LW_D, label, col, counts=[(n, col)])
    link(x_dcat, cy, x_dmid + MW_D, my, DATA, 2.8)
    ring(x_dmid, my, MW_D, 50, gname, DATA, FM)
    y = ys[-1] + PITCH + GAP
    for gname, counts in data[1:]:                        # ASAP7, NS3K: leaves at the middle level
        link(x_dcat, cy, x_dmid + MW_D, y, DATA, 2.8)
        leaf(x_dmid, y, MW_D, gname, DATA, counts=counts)
        y += PITCH
    solid(x_dcat, cy, CW, 92, 'Data,\nas tensors', DATA, FC)
    link(CX - RX, cy, x_dcat + CW, cy, DATA, 4.5, 0.75)

    # ---- right: Naive Engine, then Evaluation
    y0 = cy - h_right / 2
    mids = []
    for g, (gname, kids) in enumerate(naive):
        ys = [y0 + (g * 3 + i) * PITCH + g * GAP for i in range(3)]
        my = ys[1]
        mids.append(my)
        for yy, (label, st) in zip(ys, kids):
            link(x_rmid + MW_R, my, x_nleaf, yy, NAIVE, 2.2)
            leaf(x_nleaf, yy, LW, label, NAIVE, st)
    ny = (mids[0] + mids[1]) / 2
    for g, (gname, _) in enumerate(naive):
        link(x_rcat + CW_R, ny, x_rmid, mids[g], NAIVE, 2.8)
        ring(x_rmid, mids[g], MW_R, 50, gname, NAIVE, FM)
    eys = [y0 + (6 + i) * PITCH + 2 * GAP for i in range(6)]
    ey = (eys[0] + eys[-1]) / 2
    for yy, (label, st) in zip(eys, ev):
        link(x_rcat + CW_R, ey, x_eleaf, yy, EVAL, 2.2)
        leaf(x_eleaf, yy, LW_E, label, EVAL, st)
    link(CX + RX, cy, x_rcat, ny, NAIVE, 4.5, 0.75)
    link(CX + RX, cy, x_rcat, ey, EVAL, 4.5, 0.75)
    solid(x_rcat, ny, CW_R, 60, 'Naive Engine', NAIVE, FC)
    solid(x_rcat, ey, CW_R, 60, 'Evaluation', EVAL, FC)

    s += links + nodes
    # ---- center: the logo's gradient
    s.append('<ellipse cx="%g" cy="%g" rx="%g" ry="%g" fill="url(#glow)"/>' % (CX, cy, RX * 2.1, RY * 2.6))
    s.append('<rect x="%g" y="%g" width="%g" height="%g" rx="%g" fill="url(#brand)" filter="url(#soft)"/>'
             % (CX - RX, cy - RY, 2 * RX, 2 * RY, RY))
    s.append(t(CX, cy + 16, 'attar', 46, K['ink'], 'bold', ls=1))

    # ---- legend
    ly = H - 58
    s.append('<line x1="50" y1="%g" x2="%g" y2="%g" stroke="#ffffff" stroke-opacity="0.08"/>' % (ly - 42, W - 50, ly - 42))
    x = 50
    for label, col in (('Placement benchmark (after_fp)', PL), ('Floorplan benchmark (before_fp)', FP)):
        s.append('<rect x="%g" y="%g" width="40" height="24" rx="12" fill="%s" fill-opacity="0.13" stroke="%s" '
                 'stroke-width="1.8"/>' % (x, ly - 12, col, col))
        s.append(t(x + 54, ly + 7, label, 20, K['text'], anchor='start'))
        x += 54 + len(label) * 10.4 + 44
    s.append('<circle cx="%g" cy="%g" r="16" fill="%s"/>' % (x + 16, ly, K['cream']))
    s.append(t(x + 16, ly + 5.5, 'n', 16, K['ink'], 'bold'))
    s.append(t(x + 42, ly + 7, 'designs in the zoo', 20, K['text'], anchor='start'))
    x = W - 50
    nodes = []
    for label in ("Won't have", 'Not yet', 'Done'):
        w = len(label) * 9.6 + 26
        x -= w
        chip(x, ly, label, label, K['text'])
        x -= 16
    s += nodes
    s.append('</svg>')
    open(os.path.join(IMG, 'roadmap.svg'), 'w').write('\n'.join(s))


if __name__ == '__main__':
    os.makedirs(IMG, exist_ok=True)
    flow(); roadmap()
    print('wrote flow.svg, roadmap.svg in', IMG)
