"""The zoo on the Hugging Face Hub: distilled designs are downloaded into the local zoo on first use.

attar.load('after_fp/ispd2005/adaptec1') fetches the design when it is not in the local zoo; attar.fetch('after_fp')
fetches a mode, attar.fetch('after_fp/ispd2005') a suite, attar.fetch() everything; attar.zoo(remote=True) lists what the hub has (its index.json). `repo` picks
another dataset repo, `revision` a branch, tag or commit.
"""
import json

from .paths import zoo_dir

REPO_ID = 'RayChang987/attar-zoo'


def _hub():
    try:
        import huggingface_hub
    except ImportError:
        raise ImportError('downloading the zoo needs huggingface_hub (pip install huggingface_hub), '
                          'or pass root= a local zoo folder') from None
    return huggingface_hub


def index(repo=REPO_ID, revision=None):
    """{'mode/suite/design': {'cells': ..., 'nets': ..., 'facts': [...], 'bytes': ...}} for every design on the hub."""
    p = _hub().hf_hub_download(repo, 'index.json', repo_type='dataset', revision=revision)
    with open(p) as f:
        return json.load(f)['designs']


def fetch(*names, root=None, repo=REPO_ID, revision=None):
    """Download designs ('after_fp/orfs_asap7/aes'), suites ('after_fp/ispd2005') or modes ('before_fp') into the zoo (`root`, default paths.zoo_dir()),
    with the shared cell libraries they reference (libraries/<name>); everything when no name is given. Files already
    there are not downloaded again. Returns the zoo folder."""
    dest = zoo_dir(root)
    allow = [n.strip('/') + '/*' for n in names] or None
    _hub().snapshot_download(repo, repo_type='dataset', revision=revision, local_dir=dest, allow_patterns=allow)
    if allow:                      # the shared cell libraries the fetched designs reference (zoo/libraries/<name>)
        import glob
        import os
        need = set()
        for n in names:
            for m in glob.glob(os.path.join(dest, n.strip('/'), '**', 'manifest.json'), recursive=True):
                with open(m) as f:
                    b = (json.load(f).get('library') or {}).get('bundle')
                if b and not os.path.exists(os.path.join(dest, 'libraries', b, 'library.safetensors')):
                    need.add(b)
        if need:
            _hub().snapshot_download(repo, repo_type='dataset', revision=revision, local_dir=dest,
                                     allow_patterns=['libraries/%s/*' % b for b in sorted(need)])
    return dest
