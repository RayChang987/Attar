"""A design is self-contained: no environment variables, no paths in manifests, the fast loop reads the IR only."""
import glob
import json
import os
import re
import tempfile
import unittest

from common import DEV, SMALL, attar, need  # noqa: F401
from attar import paths


class SelfContainedTest(unittest.TestCase):
    def test_no_environment_variables(self):
        pkg = os.path.dirname(attar.__file__)
        for f in glob.glob(os.path.join(pkg, '**', '*.py'), recursive=True):
            with open(f, encoding='utf-8') as fh:
                self.assertIsNone(re.search(r'os\.environ|getenv', fh.read()), f)

    def test_zoo_dir(self):
        with tempfile.TemporaryDirectory() as t:
            self.assertEqual(paths.zoo_dir(t), os.path.abspath(t))
            self.assertEqual(attar.zoo(root=t), [])
            with self.assertRaises(FileNotFoundError):
                attar.load('after_fp/orfs_asap7/gcd', DEV, download=False, root=t)

    def test_manifests_hold_no_paths(self):
        ms = glob.glob(os.path.join(paths.zoo_dir(), '*', '*', '*', 'manifest.json'))     # <mode>/<suite>/<design>
        if not ms:
            raise unittest.SkipTest('no local zoo')
        for m in ms:
            with open(m) as f:
                man = json.load(f)
            self.assertNotIn('/', json.dumps(man.get('source', {})), m)
            self.assertNotIn('/', json.dumps(man.get('unread_sources', {})), m)
            self.assertNotIn('run', man.get('orfs', {}), m)
            for s in man.get('snapshots', {}).values():
                self.assertNotIn('path', s, m)

    def test_a_copied_design_works_anywhere(self):
        import shutil
        d0 = need(SMALL)
        with tempfile.TemporaryDirectory() as t:
            shutil.copytree(d0.path, os.path.join(t, 'gcd'))       # the IR alone, away from every raw file
            d = attar.load(os.path.join(t, 'gcd'), DEV, download=False).restore('reference')
            self.assertAlmostEqual(d.hpwl().item(), d0.restore('reference').hpwl().item(), delta=1e-3)
            self.assertIn('export', d.capabilities())
            self.assertTrue(d.tracks())

    def test_hub_round_trip(self):
        """tools/zoo_release.pack compresses a design and its library for the hub; attar.fetch, served by a stand-in
        for huggingface_hub, gives back every file byte for byte."""
        import shutil
        import sys
        from unittest import mock
        from attar import hub
        d0 = need(SMALL)
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))
        import zoo_release
        bundle = d0.manifest.get('library', {}).get('bundle')
        zoo = paths.zoo_dir()
        folders = [SMALL] + (['libraries/' + bundle] if bundle else [])
        with tempfile.TemporaryDirectory() as t:
            up, dest = os.path.join(t, 'hub'), os.path.join(t, 'zoo')
            zoo_release.pack(zoo, up, folders)
            served = sorted(os.path.relpath(os.path.join(dp, f), up) for dp, _, fs in os.walk(up) for f in fs)
            self.assertIn(SMALL + '/core.safetensors.zst', served)
            self.assertIn(SMALL + '/manifest.json', served)

            class FakeHub:
                list_repo_files = staticmethod(lambda repo, **kw: served)

                @staticmethod
                def hf_hub_download(repo, f, local_dir, **kw):
                    p = os.path.join(local_dir, f)
                    os.makedirs(os.path.dirname(p), exist_ok=True)
                    shutil.copy(os.path.join(up, f), p)
                    return p

            with mock.patch.object(hub, '_hub', lambda: FakeHub):
                d = attar.load(SMALL, DEV, root=dest)
            for folder in folders:
                for f in os.listdir(os.path.join(zoo, folder)):
                    with open(os.path.join(zoo, folder, f), 'rb') as a, open(os.path.join(dest, folder, f), 'rb') as b:
                        self.assertEqual(a.read(), b.read(), f)
            self.assertEqual([x for x in os.listdir(dest) if x.startswith('.')], [])      # no staging left behind
            self.assertAlmostEqual(d.restore('reference').hpwl().item(), d0.restore('reference').hpwl().item(), delta=1e-6)


if __name__ == '__main__':
    unittest.main()
