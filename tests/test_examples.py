"""The example programs run."""
import os  # noqa: F401
import sys  # noqa: F401
import unittest


from common import MEDIUM, need


class ExampleTest(unittest.TestCase):
    def test_quickstart_placement_runs(self):
        import subprocess
        name = 'after_fp/orfs_asap7/aes'
        need(name)
        here = os.path.join(os.path.dirname(__file__), '..', 'examples')
        r = subprocess.run([sys.executable, os.path.join(here, 'quickstart_placement.py'), name], capture_output=True, text=True, timeout=300)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertIn("'legal': True", r.stdout)
        self.assertIn('setup: WNS', r.stdout)                   # the timing report
        self.assertIn('GROV', r.stdout)
        self.assertTrue(os.path.exists(os.path.join(here, 'orfs_asap7_aes_placed.png')))

    def test_quickstart_floorplan_runs(self):
        import subprocess
        name = 'before_fp/orfs_asap7/aes'
        need(name)
        here = os.path.join(os.path.dirname(__file__), '..', 'examples')
        r = subprocess.run([sys.executable, os.path.join(here, 'quickstart_floorplan.py'), name], capture_output=True, text=True, timeout=300)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertIn('d.floorplan() places it; then d.export', r.stdout)
        self.assertTrue(os.path.exists(os.path.join(here, 'orfs_asap7_aes_fp.png')))

    def test_timing_driven_runs(self):
        import subprocess
        name = MEDIUM
        need(name)
        here = os.path.join(os.path.dirname(__file__), '..', 'examples')
        r = subprocess.run([sys.executable, os.path.join(here, 'timing_driven.py'), name], capture_output=True, text=True, timeout=300)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertEqual(r.stdout.count('setup: WNS'), 2)              # before and after the reweighting
        self.assertTrue(os.path.exists(os.path.join(here, 'orfs_asap7_aes_timing_driven.png')))

if __name__ == '__main__':
    unittest.main()
