#!/usr/bin/env python3
"""Render docs/images/attar_demo.gif: the README code typed into an editor, the output of a real run of it
(recorded by tools/make_demo_gif.py --record, timings included) appearing in a terminal, then the resulting plot.
usage: make_demo_gif.py --record   (runs the code on the GPU, writes docs/images/demo_run.json and hero_aes.png)
       make_demo_gif.py            (renders the GIF from that recording)"""
import io
import contextlib
import json
import os
import sys
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
IMG = os.path.join(ROOT, 'docs', 'images')
REC = os.path.join(IMG, 'demo_run.json')
DESIGN = 'orfs_asap7_2_1/cva6'
CODE = '''import attar

d = attar.load('orfs_asap7_2_1/cva6', device='cuda')
d.status()                # what it has, what it can do

d.floorplan()             # IO pins + macros
d.place()                 # global placement + legalization
d.plot('cva6.png')
d.export('cva6.def')      # back to DEF, then on to ORFS
'''


def record():
    sys.path.insert(0, ROOT)
    import torch
    import attar
    attar.load(DESIGN, device='cuda')                         # CUDA context warm-up, not recorded
    steps = []

    def step(name, fn):
        buf = io.StringIO(); t = time.time()
        with contextlib.redirect_stdout(buf):
            r = fn()
        torch.cuda.synchronize()
        steps.append({'step': name, 'out': buf.getvalue(), 'sec': time.time() - t})
        return r
    d = step('load', lambda: attar.load(DESIGN, device='cuda'))
    step('status', lambda: d.status())
    step('floorplan', lambda: d.floorplan())
    step('place', lambda: d.place())
    step('plot', lambda: d.plot(os.path.join(IMG, 'hero_cva6.png'), dpi=220))    # sharp enough for the GIF pane
    step('export', lambda: print(d.export(os.path.join('/tmp', 'attar_demo_cva6.def'))))
    json.dump(steps, open(REC, 'w'), indent=1)


def terminal_lines(steps):
    """(text, color, event) lines: the real output, the status block trimmed to its key lines. event 'plot' marks
    where the plot appears in the output pane."""
    S = {s['step']: s for s in steps}
    G, R, W, D, Y, B = '#5f7f1e', '#b8323f', '#2b211c', '#a39486', '#b8650f', '#8e4b8a'     # warm light terminal
    L = [('$ python quickstart.py', B, None)]
    st = S['status']['out'].splitlines()
    L.append((st[0], W, None))
    for key in ('netlist', 'cell_geometry', 'floorplan.die', 'floorplan.io', 'rows'):
        for line in st:
            if line.strip()[2:].strip().startswith(key + ' '):
                ok = line.strip().startswith('\u2705')
                body = ' '.join(line.strip()[1:].split())
                L.append(('  %s %s' % ('\u2713' if ok else '\u2717', body), G if ok else R, None)); break
    L.append(('  -> missing floorplan.io: d.floorplan() places it', D, None))
    for line in S['floorplan']['out'].splitlines():
        if line.strip().startswith(('macros:', 'io:')):
            t = line.strip()
            if t.startswith('macros:'):
                t = t.split('(')[0] + '- ' + t.rsplit(',', 2)[-2].strip()      # "macros: 8 placed - 0 overlapping pairs"
            L.append(('  ' + t[:88], W, None))
    L.append(('  floorplan  %.2f s' % S['floorplan']['sec'], Y, None))
    p = S['place']['out'].strip().replace('place: ', '')
    L.append(('  ' + p[:88], W, None))
    L.append(('  place      %.2f s   (108k cells, legal incl. power rails)' % S['place']['sec'], Y, None))
    L.append(('  plot       %.2f s   -> cva6.png' % S['plot']['sec'], Y, 'plot'))
    ex = eval(S['export']['out'].strip())
    L.append(('  export     %.2f s   -> cva6.def (%d components, %d IO pins)' % (S['export']['sec'], ex['components'], ex['pins']), Y, None))
    return L


def render():
    from PIL import Image, ImageDraw, ImageFont
    from pygments import lex
    from pygments.lexers import PythonLexer
    from pygments.token import Token
    steps = json.load(open(REC))
    Wd, Ht, LW = 1600, 900, 900                          # canvas, left (editor + terminal) width
    mono = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'
    f_code = ImageFont.truetype(mono, 20); f_term = ImageFont.truetype(mono, 15); f_ui = ImageFont.truetype(mono, 14)
    BG, PANEL, BAR, TERM, OUT = '#f6f0e6', '#fdfaf5', '#ece3d5', '#f3ebdf', '#fbf8f3'     # warm paper
    col = {Token.Keyword: '#b5452f', Token.Keyword.Namespace: '#b5452f', Token.Name.Builtin: '#8e4b8a',
           Token.Literal.String: '#5f7f1e', Token.Comment: '#a39486', Token.Literal.Number: '#b8650f',
           Token.Operator: '#b5452f', Token.Name.Namespace: '#c2501f', Token.Name.Function: '#8e4b8a'}

    def color(tt):
        while tt is not None:
            if tt in col:
                return col[tt]
            tt = tt.parent
        return '#2b211c'
    chars = []                                           # (char, color) in typing order
    for tt, v in lex(CODE, PythonLexer()):
        for ch in v:
            chars.append((ch, color(tt)))
    while chars and chars[-1][0] == '\n':
        chars.pop()
    term = terminal_lines(steps)
    code_y0, lh = 66, 29
    term_y0 = code_y0 + lh * (CODE.count('\n') + 1) + 20
    # the output pane: the plot, fitted inside it
    ox0, oy0, ox1, oy1 = LW + 14, 70, Wd - 14, Ht - 14
    plot = Image.open(os.path.join(IMG, 'hero_cva6.png')).convert('RGB')
    sc = min((ox1 - ox0) / plot.width, (oy1 - oy0) / plot.height)
    plot = plot.resize((int(plot.width * sc), int(plot.height * sc)), Image.LANCZOS)
    px, py = ox0 + ((ox1 - ox0) - plot.width) // 2, oy0 + ((oy1 - oy0) - plot.height) // 2

    def base(n_chars, n_term=0, cursor=True, plot_alpha=0.0):
        im = Image.new('RGB', (Wd, Ht), BG); dr = ImageDraw.Draw(im)
        dr.rectangle([0, 0, Wd, 40], fill=BAR)
        for k, c in enumerate(('#ff5f56', '#ffbd2e', '#27c93f')):
            dr.ellipse([16 + 24 * k, 13, 30 + 24 * k, 27], fill=c)
        dr.text((110, 12), 'quickstart.py  -  attar', font=f_ui, fill='#8a7b6d')
        dr.rectangle([0, 40, LW, term_y0 - 10], fill=PANEL)
        x, y = 28, code_y0
        for ch, c in chars[:n_chars]:
            if ch == '\n':
                x, y = 28, y + lh; continue
            dr.text((x, y), ch, font=f_code, fill=c); x += f_code.getlength(ch)
        if cursor:
            dr.rectangle([x + 2, y + 3, x + 12, y + 24], fill='#d9622b')
        dr.rectangle([0, term_y0 - 10, LW, Ht], fill=TERM)
        dr.text((LW - 100, term_y0 - 4), 'terminal', font=f_ui, fill='#b9aa98')
        for j, (t, c, _) in enumerate(term[:n_term]):
            dr.text((20, term_y0 + 16 + j * 22), t, font=f_term, fill=c)
        dr.rectangle([LW, 40, Wd, Ht], fill=OUT)
        dr.line([LW, 40, LW, Ht], fill=BAR, width=2)
        dr.text((LW + 18, 48), 'cva6.png', font=f_ui, fill='#8a7b6d')
        if plot_alpha <= 0:
            msg = "d.plot('cva6.png') draws here"
            dr.text(((ox0 + ox1 - f_ui.getlength(msg)) / 2, (oy0 + oy1) / 2), msg, font=f_ui, fill='#cdbfae')
        else:
            region = im.crop((px, py, px + plot.width, py + plot.height))
            im.paste(Image.blend(region, plot, plot_alpha), (px, py))
        return im
    frames, durs = [], []
    n = 0
    while n < len(chars):                                # typing
        n = min(len(chars), n + 4)
        frames.append(base(n)); durs.append(40)
    frames.append(base(n, cursor=False)); durs.append(700)
    shown = 0.0
    for k in range(1, len(term) + 1):                    # output, line by line; the plot fades in at its line
        if term[k - 1][2] == 'plot':
            for a in (0.15, 0.3, 0.5, 0.7, 0.85, 1.0):
                frames.append(base(n, k, cursor=False, plot_alpha=a)); durs.append(55)
            shown = 1.0
            continue
        frames.append(base(n, k, cursor=False, plot_alpha=shown)); durs.append(420 if k > 1 else 600)
    durs[-1] = 4500
    # one palette for every frame (no per-frame palettes: text stays crisp, colors do not flicker)
    ref = Image.new('RGB', (Wd, Ht * 2)); ref.paste(frames[-1], (0, 0)); ref.paste(frames[len(chars) // 4], (0, Ht))
    pal = ref.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    out = os.path.join(IMG, 'attar_demo.gif')
    q[0].save(out, save_all=True, append_images=q[1:], duration=durs, loop=0, optimize=False)
    print('wrote %s: %dx%d, %d frames, %.0f KB, %.1f s' % (out, Wd, Ht, len(frames), os.path.getsize(out) / 1e3, sum(durs) / 1e3))


if __name__ == '__main__':
    if '--record' in sys.argv:
        record()
    render()
