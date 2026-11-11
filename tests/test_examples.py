"""The example programs run."""
import os  # noqa: F401
import sys  # noqa: F401
import unittest


from common import need


class ExampleTest(unittest.TestCase):
    def test_quickstart_runs(self):
        import subprocess
        name = 'after_fp/orfs_asap7/aes'
        need(name)
        here = os.path.join(os.path.dirname(__file__), '..', 'examples')
        r = subprocess.run([sys.executable, os.path.join(here, 'quickstart.py'), name], capture_output=True, text=True, timeout=300)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertIn("'legal': True", r.stdout)
        self.assertIn('orfs_roundtrip.py', r.stdout)
        self.assertTrue(os.path.exists(os.path.join(here, 'orfs_asap7_aes_placed.png')))

if __name__ == '__main__':
    unittest.main()
