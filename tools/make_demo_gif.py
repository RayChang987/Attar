#!/usr/bin/env python3
"""Render docs/images/demo.gif: the README code typed into an editor and run on two designs. For each, the output of a
real run (recorded by tools/make_demo_gif.py --record, timings included) appears in a terminal while the plot pane
shows the floorplan as loaded, then the placement.
usage: make_demo_gif.py --record   (runs the code on the GPU, writes docs/images/demo_run.json and docs/images/demo/)
       make_demo_gif.py            (renders the GIF from that recording)"""
import io
import contextlib
import json
import os
import sys
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
IMG = os.path.join(ROOT, 'docs', 'images')
PANES = os.path.join(IMG, 'demo')                      # the plots the GIF shows, as the code wrote them
REC = os.path.join(IMG, 'demo_run.json')
CASES = [('cva6', 'after_fp/orfs_asap7/cva6'), ('ariane136', 'after_fp/ns3k/ariane136')]
CODE = '''import attar

d = attar.load('%s', device='cuda')
d.status()                   # what it has, what it can do
d.plot('floorplan.png')      # die, macros, IO pins, PDN: given

d.place()                    # global placement + legalization
d.plot('placed.png')
d.export('placed.def')       # a DEF, back to the slow loop
'''
LINE = {'load': 2, 'status': 3, 'floorplan': 4, 'place': 6, 'placed': 7, 'export': 8}     # code line each step runs


def record():
    sys.path.insert(0, ROOT)
    import torch
    import attar
    os.makedirs(PANES, exist_ok=True)
    attar.load(CASES[0][1], device='cuda')                 # CUDA context warm-up, not recorded
    run = {}
    for key, name in CASES:
        steps = []

        def step(s, fn):
            buf = io.StringIO(); t = time.time()
            with contextlib.redirect_stdout(buf):
                r = fn()
            torch.cuda.synchronize()
            steps.append({'step': s, 'out': buf.getvalue(), 'sec': time.time() - t})
            return r
        plot = lambda d, what: d.plot(os.path.join(PANES, '%s_%s.png' % (key, what))).clf()
        d = step('load', lambda: attar.load(name, device='cuda'))
        step('status', lambda: d.status())
        step('floorplan', lambda: plot(d, 'floorplan'))
        step('place', lambda: d.place())
        step('placed', lambda: plot(d, 'placed'))
        step('export', lambda: print(d.export(os.path.join('/tmp', 'attar_demo_%s.def' % key))))
        if key == 'cva6':                                   # the README gallery's cva6 is this placement
            d.plot(os.path.join(IMG, 'gallery', 'cva6.png'), dpi=220).clf()
        run[key] = steps
        del d
    json.dump(run, open(REC, 'w'), indent=1)


def terminal_lines(steps):
    """(text, color, step) lines: the real output, the status block trimmed to its key lines. step: the code step that
    printed the line (its code line is highlighted, 'floorplan'/'placed' make the plot appear)."""
    S = {s['step']: s for s in steps}
    G, R, W, D, Y, B = '#5f7f1e', '#b8323f', '#2b211c', '#a39486', '#b8650f', '#8e4b8a'     # warm light terminal
    L = [('$ python quickstart.py', B, 'load')]
    st = S['status']['out'].splitlines()
    L.append((st[0], W, 'status'))
    for line in st:
        if 'placement target density' in line:
            L.append(('  ' + line.strip().split(' (')[0], W, 'status'))
    for key in ('netlist', 'cell_geometry', 'floorplan.macros', 'pdn', 'placement'):
        for line in st:
            if line.strip()[2:].strip().startswith(key + ' '):
                ok = line.strip().startswith('✅')
                body = ' '.join(line.strip()[1:].split()).split('  (')[0]
                L.append(('  %s %s' % ('✓' if ok else '✗', body), G if ok else R, 'status')); break
    L.append(('  ' + [l for l in st if l.strip().startswith('->')][-1].strip().split(':')[0], D, 'status'))
    L.append(('  plot       %.2f s   -> floorplan.png' % S['floorplan']['sec'], Y, 'floorplan'))
    p = S['place']['out'].strip().replace('place: ', '')
    L.append(('  ' + p[:88], W, 'place'))
    ex = eval(S['export']['out'].strip())
    L.append(('  place      %.2f s   (%dk cells, legal incl. power rails, PDN)' % (S['place']['sec'], ex['components'] // 1000),
              Y, 'place'))
    L.append(('  plot       %.2f s   -> placed.png' % S['placed']['sec'], Y, 'placed'))
    L.append(('  export     %.2f s   -> placed.def (%d cells, %d nets)' % (S['export']['sec'], ex['components'], ex['nets']),
              Y, 'export'))
    return L


def render():
    from PIL import Image, ImageDraw, ImageFont
    from pygments import lex
    from pygments.lexers import PythonLexer
    from pygments.token import Token
    run = json.load(open(REC))
    Wd, Ht, LW = 1600, 900, 900                          # canvas, left (editor + terminal) width
    mono = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'
    f_code = ImageFont.truetype(mono, 20); f_term = ImageFont.truetype(mono, 15); f_ui = ImageFont.truetype(mono, 14)
    BG, PANEL, BAR, TERM, OUT, RUN = '#f6f0e6', '#fdfaf5', '#ece3d5', '#f3ebdf', '#fbf8f3', '#f6e7d6'   # warm paper
    col = {Token.Keyword: '#b5452f', Token.Keyword.Namespace: '#b5452f', Token.Name.Builtin: '#8e4b8a',
           Token.Literal.String: '#5f7f1e', Token.Comment: '#a39486', Token.Literal.Number: '#b8650f',
           Token.Operator: '#b5452f', Token.Name.Namespace: '#c2501f', Token.Name.Function: '#8e4b8a'}

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

    def panes(*paths):
        """The plots, cropped to the box their content shares (margins trimmed, so a cross-fade stays aligned)."""
        from PIL import ImageChops
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

    def base(chars, n_chars, term=(), active=None, cursor=True, plot=None, title='placed.png', caret=None):
        """plot: None (placeholder), an image, or (from, to, alpha) for a cross-fade. caret: the cursor's char index
        (default: after the last char)."""
        im = Image.new('RGB', (Wd, Ht), BG); dr = ImageDraw.Draw(im)
        dr.rectangle([0, 0, Wd, 40], fill=BAR)
        for k, c in enumerate(('#ff5f56', '#ffbd2e', '#27c93f')):
            dr.ellipse([16 + 24 * k, 13, 30 + 24 * k, 27], fill=c)
        dr.text((110, 12), 'quickstart.py  -  attar', font=f_ui, fill='#8a7b6d')
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
        dr.text((LW + 18, 48), title, font=f_ui, fill='#8a7b6d')
        if plot is None:
            msg = "d.plot(...) draws here"
            dr.text(((ox0 + ox1 - f_ui.getlength(msg)) / 2, (oy0 + oy1) / 2), msg, font=f_ui, fill='#cdbfae')
        elif len(plot) == 3:
            (a, xy), (b, _), alpha = plot
            im.paste(Image.blend(a, b, alpha), xy)
        else:
            im.paste(plot[0], plot[1])
        return im

    frames, durs = [], []

    def add(im, ms):
        frames.append(im); durs.append(ms)
    prev, prev_code, prev_term = None, None, ()          # the plot on screen, the code in the editor, its output
    for i, (key, name) in enumerate(CASES):
        code = CODE % name
        chars = chars_of(code)
        fp, pl = panes(os.path.join(PANES, key + '_floorplan.png'), os.path.join(PANES, key + '_placed.png'))
        term = terminal_lines(run[key])
        if prev_code is None:                            # type the code
            n = 0
            while n < len(chars):
                n = min(len(chars), n + 4)
                add(base(chars, n), 40)
        else:                                            # same code, another design: retype the name
            old = chars_of(prev_code)
            at = prev_code.index(CASES[i - 1][1])
            end = at + len(CASES[i - 1][1])
            for k in range(len(CASES[i - 1][1]) + 1):    # select-and-delete, one char at a time from the end
                cs = old[:end - k] + old[end:]
                add(base(cs, len(cs), prev_term, plot=prev, title='placed.png', caret=end - k), 30)
                if k == 0:
                    durs[-1] = 600
            for k in range(1, len(name) + 1):
                cs = chars[:at + k] + chars[at + len(name):]
                add(base(cs, len(cs), prev_term, plot=prev, title='placed.png', caret=at + k), 45)
        n = len(chars)
        add(base(chars, n, cursor=False, plot=prev), 700)
        shown, title = prev, 'placed.png'
        for k in range(1, len(term) + 1):                # output, line by line; plots fade in at their line
            stp = term[k - 1][2]
            if stp in ('floorplan', 'placed'):
                src, dst = (shown if shown is not None else None), (fp if stp == 'floorplan' else pl)
                title = stp + '.png'
                for a in (0.15, 0.3, 0.5, 0.7, 0.85, 1.0):
                    if src is None:
                        blank = Image.new('RGB', dst[0].size, OUT)
                        p = ((blank, dst[1]), dst, a)
                    else:
                        p = (src, dst, a)
                    add(base(chars, n, term[:k], stp, cursor=False, plot=p, title=title), 60)
                shown = dst
                durs[-1] = 2200 if stp == 'floorplan' else 900
                continue
            if stp == 'place' and term[k - 2][2] != 'place':
                for dots in ('placing', 'placing .', 'placing . .', 'placing . . .'):   # the place call runs
                    add(base(chars, n, term[:k - 1] + [('  ' + dots, '#a39486', None)], 'place', cursor=False, plot=shown,
                             title=title), 300)
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
    out = os.path.join(IMG, 'demo.gif')
    q[0].save(out, save_all=True, append_images=q[1:], duration=durs, loop=0, optimize=False)
    print('wrote %s: %dx%d, %d frames, %.0f KB, %.1f s' % (out, Wd, Ht, len(frames), os.path.getsize(out) / 1e3, sum(durs) / 1e3))


if __name__ == '__main__':
    if '--record' in sys.argv:
        record()
    render()
