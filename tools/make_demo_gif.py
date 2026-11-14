#!/usr/bin/env python3
"""Render the README demos: code typed into an editor and run, its real output (recorded with --record, timings
included) in a terminal, the plots it writes in the pane on the right.

  placement   docs/images/demo_placement.gif: timing-driven placement written by the user (examples/timing_driven.py)
              on two ASAP7 designs (cva6, then swerv): place for wirelength, weight the nets by their slack as OpenROAD's
              gpl does, place again
  floorplan   docs/images/demo_floorplan.gif: before_fp/gsrc_hard/n100 at 85% block area: d.floorplan() (GP + LP
              legalization: macros stick out, not legal), then d.floorplan(..., packing=True) (B*-tree: legal)

usage: make_demo_gif.py [placement|floorplan|all] [--record]
       --record runs the code on the GPU and writes docs/images/demo_<demo>.json and the plots in docs/images/demo/"""
import contextlib
import io
import json
import os
import sys
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
IMG = os.path.join(ROOT, 'docs', 'images')
PANES = os.path.join(IMG, 'demo')                      # the plots the GIFs show, as the code wrote them

G, R, W, D, Y, B = '#5f7f1e', '#b8323f', '#2b211c', '#a39486', '#b8650f', '#8e4b8a'     # warm light terminal

DEMOS = {
    'placement': dict(
        file='timing_driven.py',
        code='''import attar
d = attar.load('%s', device='cuda')
d.place()                                 # wirelength only
d.plot('wirelength.png')
d.timing_report(endpoints=0, path=False)
s = d.timing().net_slack()                # every net, as OpenSTA
v = s[s.isfinite()].sort().values
v = v[:len(v) // 10]                      # OpenROAD gpl: worst 10%%
lo, hi = v[len(v) // 50], v[-1]           # and its weight ramp
w = d.n_weight
d.n_weight = w * (1 + 4 * ((hi - s) / (hi - lo)).clamp(0, 1))
d.place(verbose=False)                    # timing-driven
d.n_weight = w
d.timing_report(endpoints=0, path=False)
d.plot('timing_driven.png')
''',
        cases=[('cva6', 'after_fp/orfs_asap7/cva6'), ('swerv', 'after_fp/orfs_asap7/swerv')],
        line={'load': 1, 'place': 2, 'wirelength': 3, 'report': 4, 'slack': 5, 'weights': 10, 'td': 11,
              'report_td': 13, 'timing_driven': 14},
        plots={'wirelength': 'wirelength.png', 'timing_driven': 'timing_driven.png'},      # step -> the file it writes
        busy=('td', 'placing'),
        crop_together=False),
    'floorplan': dict(
        file='quickstart_floorplan.py',
        code='''import attar

d = attar.load('before_fp/gsrc_hard/n100', device='cuda')
d.floorplan(utilization=0.85%s)   # die, IO pins, macros
print(d.floorplan_legal())        # in the outline, apart?
d.plot('floorplan.png')           # with the connections
''',
        cases=[('gplp', ''), ('packed', ', packing=True')],
        line={'load': 2, 'floorplan': 3, 'legal': 4, 'plot': 5},
        plots={'plot': 'floorplan.png'},
        busy=('floorplan', 'floorplanning'),
        crop_together=True),
}


# playback speed: every frame's time is divided by it (never below 20 ms: browsers slow shorter frames to 100 ms)
SPEED = 1.5
TYPE_CHARS = 6                                           # characters typed per frame


def rec_path(demo):
    return os.path.join(IMG, 'demo_%s.json' % demo)


def record(demo):
    sys.path.insert(0, ROOT)
    import torch
    import attar
    cfg = DEMOS[demo]
    os.makedirs(PANES, exist_ok=True)
    first = cfg['cases'][0][1] if demo == 'placement' else 'before_fp/gsrc_hard/n100'
    attar.load(first, device='cuda')                       # CUDA context warm-up, not recorded
    run = {}

    for key, arg in cfg['cases']:
        steps = []

        def step(s, fn):
            buf = io.StringIO(); t = time.time()
            with contextlib.redirect_stdout(buf):
                r = fn()
            torch.cuda.synchronize()
            steps.append({'step': s, 'out': buf.getvalue(), 'sec': time.time() - t})
            return r
        plot = lambda d, what: d.plot(os.path.join(PANES, '%s_%s.png' % (key, what))).clf()
        if demo == 'placement':                            # the code of examples/timing_driven.py, step by step
            d = step('load', lambda: attar.load(arg, device='cuda'))
            step('place', lambda: d.place())
            h0 = float(d.hpwl())
            step('wirelength', lambda: plot(d, 'wirelength'))
            step('report', lambda: d.timing_report(endpoints=0, path=False))
            s = step('slack', lambda: d.timing().net_slack())
            steps[-1]['res'] = {'nets': int(torch.isfinite(s).sum()), 'negative': int((s < 0).sum())}

            def weights():
                v = s[s.isfinite()].sort().values
                v = v[:len(v) // 10]
                lo, hi = v[len(v) // 50], v[-1]
                w = d.n_weight
                d.n_weight = w * (1 + 4 * ((hi - s) / (hi - lo)).clamp(0, 1))
                return w, len(v)
            w, nw = step('weights', weights)
            steps[-1]['res'] = {'weighted': nw}
            step('td', lambda: d.place(verbose=False))
            d.n_weight = w
            steps[-1]['res'] = {'legal': bool(attar.Evaluator(d).legality()['legal']), 'hpwl': float(d.hpwl()),
                                'hpwl0': h0}
            step('report_td', lambda: d.timing_report(endpoints=0, path=False))
            step('timing_driven', lambda: plot(d, 'timing_driven'))
        else:
            name = 'before_fp/gsrc_hard/n100'
            d = step('load', lambda: attar.load(name, device='cuda'))
            r = step('floorplan', lambda: d.floorplan(utilization=0.85, packing=bool(arg)))
            steps[-1]['res'] = {k: v for k, v in r['macros'].items() if isinstance(v, (int, float, str, bool))}
            steps[-1]['res']['hpwl'] = float(d.hpwl(d.pos))
            steps[-1]['res']['die'] = [round(float(v), 1) for v in d.die_box[0]]
            steps[-1]['res']['counts'] = {k: d.manifest['counts'][k] for k in ('macros', 'nets')}
            steps[-1]['res']['io'] = len(d.io)
            step('legal', lambda: print(d.floorplan_legal()))
            step('plot', lambda: plot(d, 'floorplan'))
        run[key] = steps
        del d
    json.dump(run, open(rec_path(demo), 'w'), indent=1)


def terminal_placement(steps, key):
    """(text, color, step) lines: the real output, the timing reports trimmed to their setup / hold lines. step: the
    code step that printed the line (its code line is highlighted; plot steps make the plot appear)."""
    S = {s['step']: s for s in steps}
    L = [('$ python timing_driven.py', B, 'load')]
    L.append(('  ' + S['place']['out'].strip().replace('place: ', '')[:88], W, 'place'))
    L.append(('  plot       %.2f s   -> wirelength.png' % S['wirelength']['sec'], Y, 'wirelength'))

    def report(stp):
        for line in S[stp]['out'].splitlines():
            t = line.strip()
            if t.startswith('setup:'):
                L.append(('  ' + t.split(' (worst')[0], R if '-' in t.split(',')[0] else G, stp))
            elif t.startswith('hold:'):
                L.append(('  ' + t, D, stp))
    report('report')
    r = S['slack']['res']
    L.append(('  net_slack  %.3f s  -> %d nets timed, %d with negative slack' % (S['slack']['sec'], r['nets'], r['negative']),
              Y, 'slack'))
    L.append(('  weights    1 .. 5 on the worst %d nets (10%%)' % S['weights']['res']['weighted'], Y, 'weights'))
    r = S['td']['res']
    L.append(('  place      %.2f s   timing-driven, legal %s, HPWL %.4g (%+.1f%%)'
              % (S['td']['sec'], r['legal'], r['hpwl'], 100 * (r['hpwl'] / r['hpwl0'] - 1)), Y, 'td'))
    report('report_td')
    L.append(('  plot       %.2f s   -> timing_driven.png' % S['timing_driven']['sec'], Y, 'timing_driven'))
    return L


def terminal_floorplan(steps, key):
    S = {s['step']: s for s in steps}
    r = S['floorplan']['res']
    die = r['die']
    L = [('$ python quickstart_floorplan.py', B, 'load'),
         ('  load       %.2f s   n100 (GSRC): %d macros, %d IO pins, %d nets, no die'
          % (S['load']['sec'], r['counts']['macros'], r['io'], r['counts']['nets']), Y, 'load'),
         ('  floorplan  %.2f s   die %g x %g (85%% block area), IO pins, macros by %s'
          % (S['floorplan']['sec'], die[2] - die[0], die[3] - die[1],
             'B*-tree packing' if key == 'packed' else 'GP + LP legalization'), Y, 'floorplan'),
         ('  macros: %d overlapping pairs, %s, HPWL %s'
          % (r['overlapping_pairs'], '%d outside the core' % r['outside_core'] if r['outside_core'] else 'all in the core',
             '{:,.0f}'.format(r['hpwl'])), W, 'floorplan')]
    legal = S['legal']['out'].strip()
    L.append((legal, G if legal.startswith('(True') else R, 'legal'))
    L.append(('  plot       %.2f s   -> floorplan.png' % S['plot']['sec'], Y, 'plot'))
    return L


TERMINAL = {'placement': terminal_placement, 'floorplan': terminal_floorplan}


def render(demo):
    from PIL import Image, ImageDraw, ImageFont, ImageChops
    from pygments import lex
    from pygments.lexers import PythonLexer
    from pygments.token import Token
    cfg = DEMOS[demo]
    run = json.load(open(rec_path(demo)))
    CODE, CASES, LINE, PLOTS = cfg['code'], cfg['cases'], cfg['line'], cfg['plots']
    Wd, Ht, LW = 1600, 900, 900                          # canvas, left (editor + terminal) width
    mono = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'
    f_code = ImageFont.truetype(mono, 20); f_term = ImageFont.truetype(mono, 15); f_ui = ImageFont.truetype(mono, 14)
    BG, PANEL, BAR, TERM, OUT, RUN = '#f6f0e6', '#fdfaf5', '#ece3d5', '#f3ebdf', '#fbf8f3', '#f6e7d6'   # warm paper
    col = {Token.Keyword: '#b5452f', Token.Keyword.Namespace: '#b5452f', Token.Name.Builtin: '#8e4b8a',
           Token.Literal.String: '#5f7f1e', Token.Comment: '#a39486', Token.Literal.Number: '#b8650f',
           Token.Operator: '#b5452f', Token.Name.Namespace: '#c2501f', Token.Name.Function: '#8e4b8a',
           Token.Keyword.Constant: '#b8650f'}

    def color(tt):
        while tt is not None:
            if tt in col:
                return col[tt]
            tt = tt.parent
        return '#2b211c'

    def chars_of(code):                                  # (char, color) in typing order
        cs = []
        for tt, v in lex(code, PythonLexer()):
            for ch in v:
                cs.append((ch, color(tt)))
        while cs and cs[-1][0] == '\n':
            cs.pop()
        return cs
    code_y0, lh = 66, 29
    term_y0 = code_y0 + lh * (CODE.count('\n') + 1) + 20
    ox0, oy0, ox1, oy1 = LW + 14, 70, Wd - 14, Ht - 14   # the output pane: the plot, fitted inside it

    def crop(paths):
        """The plots, cropped to the box their content shares (margins trimmed, so a cross-fade stays aligned)."""
        ims = [Image.open(p).convert('RGB') for p in paths]
        boxes = [ImageChops.difference(im, Image.new('RGB', im.size, im.getpixel((0, 0)))).getbbox() for im in ims]
        x0, y0 = min(b[0] for b in boxes) - 16, min(b[1] for b in boxes) - 16
        x1, y1 = max(b[2] for b in boxes) + 16, max(b[3] for b in boxes) + 16
        return [pane(im.crop((max(0, x0), max(0, y0), min(im.width, x1), min(im.height, y1)))) for im in ims]

    def pane(im):
        sc = min((ox1 - ox0) / im.width, (oy1 - oy0) / im.height)
        im = im.resize((int(im.width * sc), int(im.height * sc)), Image.LANCZOS)
        full = Image.new('RGB', (ox1 - ox0, oy1 - oy0), im.getpixel((0, 0)))   # pane-sized: plots of any size fade
        full.paste(im, (((ox1 - ox0) - im.width) // 2, ((oy1 - oy0) - im.height) // 2))
        return full, (ox0, oy0)

    path = lambda key, stp: os.path.join(PANES, '%s_%s.png' % (key, PLOTS[stp].split('.')[0]))
    if cfg['crop_together']:
        allp = [(k, s) for k, _ in CASES for s in PLOTS]
        P = dict(zip(allp, crop([path(k, s) for k, s in allp])))
    else:
        P = {}
        for k, _ in CASES:
            P.update(zip([(k, s) for s in PLOTS], crop([path(k, s) for s in PLOTS])))

    def base(chars, n_chars, term=(), active=None, cursor=True, plot=None, title=None, caret=None):
        """plot: None (placeholder), an image, or (from, to, alpha) for a cross-fade. caret: the cursor's char index
        (default: after the last char)."""
        im = Image.new('RGB', (Wd, Ht), BG); dr = ImageDraw.Draw(im)
        dr.rectangle([0, 0, Wd, 40], fill=BAR)
        for k, c in enumerate(('#ff5f56', '#ffbd2e', '#27c93f')):
            dr.ellipse([16 + 24 * k, 13, 30 + 24 * k, 27], fill=c)
        dr.text((110, 12), '%s  -  attar' % cfg['file'], font=f_ui, fill='#8a7b6d')
        dr.rectangle([0, 40, LW, term_y0 - 10], fill=PANEL)
        if active is not None:                           # the line the run is at
            y = code_y0 + lh * LINE[active]
            dr.rectangle([0, y - 3, LW, y + lh - 3], fill=RUN)
            dr.rectangle([0, y - 3, 4, y + lh - 3], fill='#d9622b')
        x, y = 28, code_y0
        caret = n_chars if caret is None else caret
        cx, cy = x, y
        for i, (ch, c) in enumerate(chars[:n_chars]):
            if ch == '\n':
                x, y = 28, y + lh
            else:
                dr.text((x, y), ch, font=f_code, fill=c); x += f_code.getlength(ch)
            if i + 1 == caret:
                cx, cy = x, y
        if cursor:
            dr.rectangle([cx + 2, cy + 3, cx + 4, cy + 24], fill='#d9622b') if caret < n_chars else \
                dr.rectangle([cx + 2, cy + 3, cx + 12, cy + 24], fill='#d9622b')
        dr.rectangle([0, term_y0 - 10, LW, Ht], fill=TERM)
        dr.text((LW - 100, term_y0 - 4), 'terminal', font=f_ui, fill='#b9aa98')
        for j, (t, c, _) in enumerate(term):
            dr.text((20, term_y0 + 16 + j * 22), t, font=f_term, fill=c)
        dr.rectangle([LW, 40, Wd, Ht], fill=OUT)
        dr.line([LW, 40, LW, Ht], fill=BAR, width=2)
        dr.text((LW + 18, 48), title or list(PLOTS.values())[-1], font=f_ui, fill='#8a7b6d')
        if plot is None:
            msg = "d.plot(...) draws here"
            dr.text(((ox0 + ox1 - f_ui.getlength(msg)) / 2, (oy0 + oy1) / 2), msg, font=f_ui, fill='#cdbfae')
        elif len(plot) == 3:
            (a, xy), (b_, _), alpha = plot
            im.paste(Image.blend(a, b_, alpha), xy)
        else:
            im.paste(plot[0], plot[1])
        return im

    frames, durs = [], []

    def add(im, ms):
        frames.append(im); durs.append(ms)
    busy_step, busy_word = cfg['busy']
    prev, prev_code, prev_term, title = None, None, (), None   # the plot on screen, the code, its output, the pane title
    for i, (key, arg) in enumerate(CASES):
        code = CODE % arg
        chars = chars_of(code)
        term = TERMINAL[demo](run[key], key)
        if prev_code is None:                            # type the code
            n = 0
            while n < len(chars):
                n = min(len(chars), n + TYPE_CHARS)
                add(base(chars, n), 40)
        else:                                            # the next case: delete what differs, type the new text
            old = chars_of(prev_code)
            a = 0
            while a < min(len(old), len(chars)) and old[a][0] == chars[a][0]:
                a += 1
            z = 0
            while z < min(len(old), len(chars)) - a and old[-1 - z][0] == chars[-1 - z][0]:
                z += 1
            end_old, end_new = len(old) - z, len(chars) - z
            for k in range(end_old - a + 1):             # delete one char at a time from the end of the old text
                cs = old[:end_old - k] + old[end_old:]
                add(base(cs, len(cs), prev_term, plot=prev, title=title, caret=end_old - k), 30)
                if k == 0:
                    durs[-1] = 600
            for k in range(1, end_new - a + 1):
                cs = chars[:a + k] + chars[end_new:]
                add(base(cs, len(cs), prev_term, plot=prev, title=title, caret=a + k), 45)
        n = len(chars)
        add(base(chars, n, cursor=False, plot=prev, title=title), 700)
        shown = prev
        for k in range(1, len(term) + 1):                # output, line by line; plots fade in at their line
            stp = term[k - 1][2]
            if stp in PLOTS:
                dst = P[(key, stp)]
                title = PLOTS[stp]
                for al in (0.15, 0.3, 0.5, 0.7, 0.85, 1.0):
                    src = shown if shown is not None else (Image.new('RGB', dst[0].size, OUT), dst[1])
                    add(base(chars, n, term[:k], stp, cursor=False, plot=(src, dst, al), title=title), 60)
                shown = dst
                durs[-1] = 2200 if stp == list(PLOTS)[0] else 900
                continue
            if stp == busy_step and term[k - 2][2] != busy_step:
                for dots in ('', ' .', ' . .', ' . . .'):   # the call runs
                    add(base(chars, n, term[:k - 1] + [('  ' + busy_word + dots, D, None)], busy_step, cursor=False,
                             plot=shown, title=title), 300)
            add(base(chars, n, term[:k], stp, cursor=False, plot=shown, title=title), 420 if k > 1 else 600)
        add(base(chars, n, term, None, cursor=False, plot=shown, title=title), 4000)
        prev, prev_code, prev_term = shown, code, term
    # one palette for every frame (no per-frame palettes: text stays crisp, colors do not flicker)
    picks = [frames[-1], frames[len(frames) // 2], frames[20]] + [f for f, d in zip(frames, durs) if d == 2200]
    ref = Image.new('RGB', (Wd, Ht * len(picks)))
    for j, f in enumerate(picks):
        ref.paste(f, (0, Ht * j))
    pal = ref.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    out = os.path.join(IMG, 'demo_%s.gif' % demo)
    durs = [max(20, int(round(d / SPEED))) for d in durs]
    q[0].save(out, save_all=True, append_images=q[1:], duration=durs, loop=0, optimize=False)
    print('wrote %s: %dx%d, %d frames, %.0f KB, %.1f s' % (out, Wd, Ht, len(frames), os.path.getsize(out) / 1e3, sum(durs) / 1e3))


if __name__ == '__main__':
    which = [a for a in sys.argv[1:] if not a.startswith('--')]
    demos = list(DEMOS) if not which or which == ['all'] else which
    for dm in demos:
        if '--record' in sys.argv:
            record(dm)
        render(dm)
