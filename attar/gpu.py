"""GPU out-of-memory errors that say what ran out and what to do about it (attar.gpu.explain_oom)."""
import functools

import torch


class GPUOutOfMemory(torch.cuda.OutOfMemoryError):
    """CUDA ran out of memory in an Attar step: the message says which step, on which design, how much of the GPU
    was free and held by other processes, and the ways out. A torch.cuda.OutOfMemoryError, so handlers of that still
    catch it."""


def _gib(x):
    return x / 2 ** 30


def describe(what, d, err):
    """The message of a GPUOutOfMemory: what ran out, the design, the device's memory, what to do."""
    parts = []
    name = None
    if d is not None and hasattr(d, 'manifest'):
        name = d.manifest.get('design') or getattr(d, 'path', None)
        try:
            parts.append('%s, %d cells, %d pins' % (name, len(d.names['cells']), len(d.p_cell)))
        except Exception:  # noqa: BLE001
            parts.append(str(name))
    dev = getattr(d, 'device', None)
    mem = ''
    try:
        dev = torch.device(dev) if dev is not None else torch.device('cuda')
        if dev.type == 'cuda' and dev.index is None:
            dev = torch.device('cuda', torch.cuda.current_device())
        if dev.type == 'cuda':
            free, total = torch.cuda.mem_get_info(dev)
            mine = torch.cuda.memory_reserved(dev)
            other = max(0, total - free - mine)
            mem = ('%s: %.1f GiB in all, %.1f GiB free, %.1f GiB held by this process, %.1f GiB by other processes'
                   % (torch.cuda.get_device_name(dev), _gib(total), _gib(free), _gib(mine), _gib(other)))
    except Exception:  # noqa: BLE001
        pass
    lines = ['the GPU ran out of memory in %s%s.' % (what, ' (%s)' % parts[0] if parts else '')]
    if mem:
        lines.append(mem + '.')
    lines.append('Ways out: free GPU memory (stop other processes on this GPU; del designs and results you no longer '
                 'need, then torch.cuda.empty_cache()), use a GPU with more memory, or load the design on the CPU, '
                 "attar.load(name, device='cpu'): the same results, slower.")
    lines.append('CUDA said: %s' % str(err).split('\n')[0])
    return '\n'.join(lines)


def explain_oom(what):
    """Decorator: a torch.cuda.OutOfMemoryError raised inside becomes a GPUOutOfMemory that explains it (once: an
    explained error passes through outer wrappers unchanged). The design is the first argument (a method's self or a
    function's d)."""
    def wrap(fn):
        @functools.wraps(fn)
        def inner(*args, **kw):
            try:
                return fn(*args, **kw)
            except GPUOutOfMemory:
                raise
            except torch.cuda.OutOfMemoryError as e:
                d = args[0] if args else None
                raise GPUOutOfMemory(describe(what, d, e)) from e
        return inner
    return wrap
