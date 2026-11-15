"""The zoo on the Hugging Face Hub: distilled designs are downloaded into the local zoo on first use.

attar.load('after_fp/ispd2005/adaptec1') fetches the design when it is not in the local zoo; attar.fetch('after_fp')
fetches a mode, attar.fetch('after_fp/ispd2005') a suite, attar.fetch() everything; attar.zoo(remote=True) lists what the hub has (its index.json). `repo` picks
another dataset repo, `revision` a branch, tag or commit.

On the hub the arrays and large JSON files are zstd-compressed (`core.safetensors.zst`, tools/zoo_release.py); they
are decompressed on download, so the local zoo holds the plain files Design reads.
"""
import json
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor

from .paths import zoo_dir

REPO_ID = 'RayChang987/attar-zoo'
ZST = '.zst'


def _hub():
    try:
        import huggingface_hub
    except ImportError:
        raise ImportError('downloading the zoo needs huggingface_hub (pip install huggingface_hub), '
                          'or pass root= a local zoo folder') from None
    return huggingface_hub


def _zstd():
    try:
        import zstandard
    except ImportError:
        raise ImportError('the zoo on the hub is zstd-compressed: pip install zstandard') from None
    return zstandard


def index(repo=REPO_ID, revision=None):
    """{'mode/suite/design': {'cells': ..., 'nets': ..., 'facts': [...], 'bytes': ...}} for every design on the hub."""
    p = _hub().hf_hub_download(repo, 'index.json', repo_type='dataset', revision=revision)
    with open(p) as f:
        return json.load(f)['designs']


def local_name(hub_name):
    """The zoo file a hub file becomes: 'x/core.safetensors.zst' -> 'x/core.safetensors'."""
    return hub_name[:-len(ZST)] if hub_name.endswith(ZST) else hub_name


def fetch(*names, root=None, repo=REPO_ID, revision=None):
    """Download designs ('after_fp/orfs_asap7/aes'), suites ('after_fp/ispd2005') or modes ('before_fp') into the zoo (`root`, default paths.zoo_dir()),
    with the shared cell libraries they reference (libraries/<name>); everything when no name is given. Files already
    there are not downloaded again. Returns the zoo folder."""
    dest = zoo_dir(root)
    files = _hub().list_repo_files(repo, repo_type='dataset', revision=revision)
    prefixes = [n.strip('/') + '/' for n in names]
    _download([f for f in files if not prefixes or any(f.startswith(p) for p in prefixes)], dest, repo, revision)
    if prefixes:                   # the shared cell libraries the fetched designs reference (zoo/libraries/<name>)
        import glob
        need = set()
        for n in names:
            for m in glob.glob(os.path.join(dest, n.strip('/'), '**', 'manifest.json'), recursive=True):
                with open(m) as f:
                    b = (json.load(f).get('library') or {}).get('bundle')
                if b:
                    need.add('libraries/%s/' % b)
        if need:
            _download([f for f in files if any(f.startswith(p) for p in need)], dest, repo, revision)
    return dest


def _download(files, dest, repo, revision, workers=8):
    """Download the hub files whose local file is missing into dest, decompressing .zst ones. Each file lands under a
    temporary name and is renamed into place, and a design's manifest.json goes last: a design folder with a
    manifest is complete."""
    todo = sorted((f for f in files if not os.path.exists(os.path.join(dest, local_name(f)))),
                  key=lambda f: (os.path.basename(f) == 'manifest.json', f))
    if not todo:
        return
    os.makedirs(dest, exist_ok=True)
    hub = _hub()
    staging = tempfile.mkdtemp(prefix='.download-', dir=dest)
    try:
        get = lambda f: hub.hf_hub_download(repo, f, repo_type='dataset', revision=revision,  # noqa: E731
                                            local_dir=staging)
        rest = [f for f in todo if os.path.basename(f) != 'manifest.json']
        last = [f for f in todo if os.path.basename(f) == 'manifest.json']
        for batch in (rest, last):
            with ThreadPoolExecutor(workers) as ex:
                got = list(ex.map(get, batch))
            for f, p in zip(batch, got):
                out = os.path.join(dest, local_name(f))
                os.makedirs(os.path.dirname(out), exist_ok=True)
                tmp = out + '.part'
                if f.endswith(ZST):
                    with open(p, 'rb') as fi, open(tmp, 'wb') as fo:
                        _zstd().ZstdDecompressor().copy_stream(fi, fo)
                    os.remove(p)
                else:
                    shutil.move(p, tmp)
                os.replace(tmp, out)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
