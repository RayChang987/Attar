"""Minimal safetensors reader/writer (format spec: 8-byte LE header size, JSON header, raw little-endian data).

Kept dependency-free so the core installs without the `safetensors` package; files are readable by it.
"""
import json
import struct

import numpy as np

_DT = {np.dtype('int8'): 'I8', np.dtype('uint8'): 'U8', np.dtype('int16'): 'I16', np.dtype('int32'): 'I32',
       np.dtype('int64'): 'I64', np.dtype('float32'): 'F32', np.dtype('float64'): 'F64', np.dtype('bool'): 'BOOL'}
_TD = {v: k for k, v in _DT.items()}


def save(path, tensors, metadata=None):
    header, blobs, off = {}, [], 0
    for name in sorted(tensors):
        a = np.ascontiguousarray(tensors[name])
        b = a.astype(a.dtype.newbyteorder('<'), copy=False).tobytes()
        header[name] = {'dtype': _DT[a.dtype], 'shape': list(a.shape), 'data_offsets': [off, off + len(b)]}
        blobs.append(b); off += len(b)
    if metadata:
        header['__metadata__'] = {k: str(v) for k, v in metadata.items()}
    h = json.dumps(header, separators=(',', ':')).encode()
    h += b' ' * (-len(h) % 8)
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(h))); f.write(h)
        for b in blobs: f.write(b)


def load(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(n))
        data = f.read()
    meta = header.pop('__metadata__', {})
    out = {}
    for name, h in header.items():
        a, b = h['data_offsets']
        out[name] = np.frombuffer(data[a:b], dtype=_TD[h['dtype']].newbyteorder('<')).reshape(h['shape']).copy()
    return out, meta
