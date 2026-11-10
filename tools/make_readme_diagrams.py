#!/usr/bin/env python3
"""README diagrams as SVG (docs/images): flow.svg (how Attar works) and roadmap.svg (where it stands).
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
    """Soft: a thin gradient track, nodes with soft gradients and halos (no hard outlines), tinted status chips."""
    W, H = 1240, 420
    s = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="%d" height="%d">' % (W, H, W, H),
         '<defs>'
         '<linearGradient id="bg" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#fdfbf7"/>'
         '<stop offset="1" stop-color="#f8f2e9"/></linearGradient>'
         '<linearGradient id="track" gradientUnits="userSpaceOnUse" x1="210" y1="0" x2="620" y2="0"><stop offset="0" stop-color="%s"/>'
         '<stop offset="1" stop-color="%s"/></linearGradient>'
         '<radialGradient id="lit" cx="0.38" cy="0.32" r="0.75"><stop offset="0" stop-color="#ec8a57"/>'
         '<stop offset="1" stop-color="%s"/></radialGradient>'
         '<radialGradient id="halo" r="0.5"><stop offset="0.55" stop-color="%s" stop-opacity="0.28"/>'
         '<stop offset="1" stop-color="%s" stop-opacity="0"/></radialGradient>'
         '<radialGradient id="halo_a" r="0.5"><stop offset="0.55" stop-color="%s" stop-opacity="0.22"/>'
         '<stop offset="1" stop-color="%s" stop-opacity="0"/></radialGradient>'
         '<radialGradient id="dim" cx="0.38" cy="0.32" r="0.75"><stop offset="0" stop-color="#fbf6ee"/>'
         '<stop offset="1" stop-color="#efe5d6"/></radialGradient>'
         '<clipPath id="half"><rect x="0" y="0" width="0" height="0"/></clipPath>'
         '</defs>' % (C['terra'], C['amber'], C['terra'], C['terra'], C['terra'], C['amber'], C['amber']),
         '<rect width="%d" height="%d" rx="26" fill="url(#bg)"/>' % (W, H)]
    s.append(text(56, 62, 'Where Attar stands', 20, C['ink2'], 'bold', 'start'))
    s.append(text(56, 90, 'Each proxy is lit once it is fast in Attar and shown to agree with the slow loop', 15.5,
                  C['muted'], anchor='start'))
    xs, y, R = (210, 620, 1030), 172, 30
    # track: soft sand underneath, a terracotta-to-amber stroke up to the half-lit node
    s.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#ede3d4" stroke-width="6" stroke-linecap="round"/>' % (xs[0], y, xs[2], y))
    s.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="url(#track)" stroke-width="6" stroke-linecap="round"/>' % (xs[0], y, xs[1], y))
    # 1: HPWL, lit
    s.append('<circle cx="%d" cy="%d" r="%d" fill="url(#halo)"/>' % (xs[0], y, R * 2.1))
    s.append('<circle cx="%d" cy="%d" r="%d" fill="url(#lit)"/>' % (xs[0], y, R))
    s.append('<path d="M %g %g l 9 10 l 18 -21" fill="none" stroke="%s" stroke-width="4.5" stroke-linecap="round" '
             'stroke-linejoin="round"/>' % (xs[0] - 13, y + 1, C['white']))
    # 2: routed WL, half lit: a soft amber left half over a pale disc
    s.append('<circle cx="%d" cy="%d" r="%d" fill="url(#halo_a)"/>' % (xs[1], y, R * 1.9))
    s.append('<circle cx="%d" cy="%d" r="%d" fill="url(#dim)"/>' % (xs[1], y, R))
    s.append('<path d="M %g %g A %d %d 0 0 0 %g %g Z" fill="%s" opacity="0.9"/>' % (xs[1], y - R, R, R, xs[1], y + R, C['amber']))
    s.append('<circle cx="%d" cy="%d" r="%d" fill="none" stroke="%s" stroke-opacity="0.35" stroke-width="2"/>' % (xs[1], y, R, C['amber']))
    # 3: timing, not lit: a pale disc
    s.append('<circle cx="%d" cy="%d" r="%d" fill="url(#dim)"/>' % (xs[2], y, R))
    s.append('<circle cx="%d" cy="%d" r="%d" fill="none" stroke="#e3d6c3" stroke-width="2"/>' % (xs[2], y, R))
    items = [('HPWL evaluation', 'Done', ('#f9e2d3', C['terra_d']), ['Exact, on the GPU: 3 ms for 8.9M pins',
                                                                     'Equals OpenROAD to 7 digits',
                                                                     'Legality = check_placement, 9/9']),
             ('Routed wirelength evaluation', 'Half way', ('#f8ebcf', '#a0700f'), ['ORFS hand-off routes, 0 DRC',
                                                                                   'HPWL vs routed WL: 951 runs',
                                                                                   'Next: a global-routing proxy']),
             ('Timing evaluation', 'Not yet', ('#f0e8dc', C['muted']), ['Liberty + SDC recorded, not read',
                                                                        'HPWL does not predict WNS / TNS',
                                                                        'Next: a GPU timing proxy vs ORFS STA'])]
    for k, (x, (t, st, (chip, chip_ink), lines)) in enumerate(zip(xs, items)):
        on = k < 2
        s.append(text(x, y + 82, t, 21, C['ink'] if on else C['ink2'], 'bold'))
        w = len(st) * 9.5 + 36
        s.append('<rect x="%g" y="%g" width="%g" height="28" rx="14" fill="%s"/>' % (x - w / 2, y + 98, w, chip))
        s.append(text(x, y + 117, st, 14.5, chip_ink, 'bold'))
        for i, l in enumerate(lines):
            s.append(text(x, y + 158 + i * 25, l, 15, C['ink2'] if on else C['muted']))
    s.append('</svg>')
    open(os.path.join(IMG, 'roadmap.svg'), 'w').write('\n'.join(s))


if __name__ == '__main__':
    os.makedirs(IMG, exist_ok=True)
    flow(); roadmap()
    print('wrote flow.svg, roadmap.svg in', IMG)
