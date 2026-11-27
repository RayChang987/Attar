#!/usr/bin/env python3
"""README diagrams as SVG (docs/images): flow.svg (how Attar works) and roadmap.svg (the roadmap as three cards).
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
    # regions: the fast loop (Attar) and the user's own flow
    s.append('<rect x="770" y="28" width="482" height="814" rx="22" fill="%s"/>' % C['terra_l'])
    s.append(text(796, 66, 'Fast loop', 19, C['terra_d'], 'bold', 'start'))
    s.append(text(796 + 9 * 19 * 0.62 + 16, 66, 'Seconds, on the GPU', 17, C['terra_d'], anchor='start'))
    s.append('<rect x="28" y="494" width="712" height="348" rx="22" fill="%s"/>' % C['slow'])
    s.append(text(54, 532, 'Your flow', 19, C['sand_d'], 'bold', 'start'))
    s.append(text(54 + 9 * 19 * 0.62 + 16, 532, 'Any P&R tool that reads DEF', 17, C['sand_d'], anchor='start'))
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
    # export -> DEF -> the user's flow
    s.append(arrow([(810, 232), (748, 232), (748, 452), (130, 452), (130, 600)], label='Export', at=(440, 452)))
    s.append(box(44, 606, 172, 136, 'DEF', ['Placement', 'Floorplan']))
    s.append(box(282, 576, 438, 196, 'OpenROAD, Innovus, ICC2, ...', ['PDN, CTS, routing', 'Sign-off'],
                 stroke=C['sand_d'], tsize=23))
    s.append(arrow([(216, 674), (276, 674)]))
    s.append('</svg>')
    open(os.path.join(IMG, 'flow.svg'), 'w').write('\n'.join(s))


def roadmap():
    """Three flat cards in the flow diagram's palette: Data (design counts per suite), the naive engines, Evaluation.
    Sized for the README's ~830 px column, so 20 px text shows at about 16 px. Status: a check (done),
    an open circle (not yet); what Attar leaves to the full flow on the bottom line."""
    data = [('Contest benchmarks', [('ISPD 2005', 8, None), ('ISPD 2015', 20, None), ('ISPD 2018', 10, None),
                                    ('ISPD 2019', 10, None), ('ISPD 2024', 5, None), ('ICCAD 2015', 8, None),
                                    ('ICCAD 2025', 13, None), ('GSRC placement', 4, None), ('MCNC', None, 10),
                                    ('GSRC blocks', None, 12), ('IBM-HB+ (ISPD 2006)', None, 28)]),
            ('ORFS and NS3K designs', [('ASAP7', 15, 5), ('Nangate45', 8, 8), ('NS3K', 98, None)])]
    naive = [('Floorplan', [('Die sizing', 'done'), ('Packing', 'done'), ('Macro placement', 'done'),
                            ('IO placement', 'done')]),
             ('Placement', [('Global placer', 'done'), ('Legalizer', 'done'), ('PDN-aware refinement', 'done'),
                            ('Detailed placement', 'done')]),
             ('Clock tree', [('Clock tree synthesis', 'done'), ('Delay-balanced trees', 'next')]),
             ('Routing', [('Global router', 'done')]),
             ('Netlist edits', [('Gate sizing', 'done'), ('Buffering and cloning', 'done'), ('Debuffering', 'done')])]
    ev = [('Wirelength', [('HPWL', 'done'), ('Steiner wirelength', 'done')]),
          ('Placement', [('Legality', 'done'), ('Displacement', 'done'), ('Density overflow', 'done')]),
          ('Routability', [('RUDY', 'done'), ('Global routing overflow', 'done'), ('Blocked pin checking', 'done')]),
          ('Timing and power', [('Setup and hold', 'done'), ('GR parasitics', 'done'),
                                ('Propagated clocks, CRPR', 'done'), ('Slew, cap, fanout (DRV)', 'done'),
                                ('Power', 'done')])]
    out = 'PDN synthesis, detailed routing, routing DRC, sign-off'

    FS, FG, PITCH, GAP = 20, 16, 34, 18                   # row text, group label, row pitch, gap before a group
    M, G, TOP = 24, 18, 24
    widths = (400, 310, 320)
    W = M * 2 + sum(widths) + G * 2
    xs = [M, M + widths[0] + G, M + widths[0] + widths[1] + 2 * G]

    def rows_h(groups):
        return sum(GAP + PITCH + len(k) * PITCH for _, k in groups)
    HEAD = 84
    card_h = HEAD + max(rows_h(data), rows_h(naive), rows_h(ev)) + 6
    LEG = 56
    H = TOP + card_h + LEG
    s = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="%d" height="%d">' % (W, H, W, H),
         '<rect width="%d" height="%d" rx="18" fill="%s"/>' % (W, H, C['paper'])]

    def card(x, w, title, sub, mono_sub=False):
        s.append('<rect x="%g" y="%g" width="%g" height="%g" rx="12" fill="%s" stroke="%s" stroke-width="1.4"/>'
                 % (x, TOP, w, card_h, C['card'], C['line']))
        s.append(text(x + 20, TOP + 42, title, 24, C['ink'], 'bold', 'start'))
        s.append(text(x + 20, TOP + 68, sub, 16, C['ink2'], anchor='start', mono=mono_sub))

    def mark(x, y, st, col):
        if st == 'done':
            s.append('<circle cx="%g" cy="%g" r="9" fill="%s"/>' % (x, y, col))
            s.append('<path d="M %g %g l 3.2 3.4 l 6 -7" fill="none" stroke="%s" stroke-width="2.2" '
                     'stroke-linecap="round" stroke-linejoin="round"/>' % (x - 4.6, y + 0.4, C['white']))
        elif st == 'next':
            s.append('<circle cx="%g" cy="%g" r="8" fill="none" stroke="%s" stroke-width="2"/>' % (x, y, col))
        else:
            s.append('<path d="M %g %g h 12" stroke="%s" stroke-width="2.2" stroke-linecap="round"/>'
                     % (x - 6, y, C['muted']))

    def group(x, y, label):
        s.append(text(x + 20, y + FG * 0.35, label, FG, C['ink2'], 'bold', 'start'))

    # Data: one row per suite, design counts in two columns
    x, w = xs[0], widths[0]
    card(x, w, 'Data', '262 designs in the zoo, loaded as tensors')
    cp, cf = x + w - 136, x + w - 50
    y = TOP + HEAD
    for g, (gname, kids) in enumerate(data):
        y += GAP
        group(x, y, gname)
        if g == 0:                                        # column heads once, on the first group's line
            s.append(text(cp, y + 5, 'Placement', 14, C['muted']))
            s.append(text(cf, y + 5, 'Floorplan', 14, C['muted']))
        for label, p, f in kids:
            y += PITCH
            s.append(text(x + 20, y + FS * 0.35, label, FS, C['ink'], anchor='start'))
            for cx, n in ((cp, p), (cf, f)):
                if n:
                    s.append(text(cx, y + FS * 0.35, str(n), FS, C['ink'], 'bold'))
        y += PITCH

    # engines and evaluators: one row per feature with its status
    for x, w, col, (title, sub, mono), groups in (
            (xs[1], widths[1], C['terra'], ('Engines', 'attar.naive, d.edit()', True), naive),
            (xs[2], widths[2], C['amber'], ('Evaluation', 'What Attar measures', False), ev)):
        card(x, w, title, sub, mono)
        y = TOP + HEAD
        for gname, kids in groups:
            y += GAP
            group(x, y, gname)
            for label, st in kids:
                y += PITCH
                mark(x + 31, y, st, col)
                s.append(text(x + 52, y + FS * 0.35, label, FS, C['muted'] if st == 'out' else C['ink'],
                              anchor='start'))
            y += PITCH

    # legend, then what Attar leaves to the full flow
    ly = TOP + card_h + LEG / 2 + 2
    x = M + 12
    for st, label in (('done', 'Done'), ('next', 'Not yet')):
        mark(x, ly, st, C['ink2'])
        s.append(text(x + 18, ly + 6, label, 16, C['ink2'], anchor='start'))
        x += 18 + len(label) * 9 + 36
    x += 24
    mark(x + 6, ly, 'out', C['muted'])
    s.append(text(x + 22, ly + 6, 'Left to your P&R tool:', 16, C['ink2'], 'bold', 'start'))
    s.append(text(x + 22 + 22 * 9 + 12, ly + 6, out, 16, C['ink2'], anchor='start'))
    s.append('</svg>')
    open(os.path.join(IMG, 'roadmap.svg'), 'w').write('\n'.join(s))


if __name__ == '__main__':
    os.makedirs(IMG, exist_ok=True)
    flow(); roadmap()
    print('wrote flow.svg, roadmap.svg in', IMG)
