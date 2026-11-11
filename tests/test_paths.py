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
            with open(f) as fh:
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


if __name__ == '__main__':
    unittest.main()
