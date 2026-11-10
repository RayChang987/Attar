"""The example programs run."""
import os  # noqa: F401
import sys  # noqa: F401
import unittest


from common import need


class ExampleTest(unittest.TestCase):
    def test_quickstart_runs(self):
        import subprocess
        name = 'orfs_asap7_2_1/aes'
        need(name)
        here = os.path.join(os.path.dirname(__file__), '..', 'examples')
        template = os.path.join(here, '..', 'orfs_runs', 'rt', 'aes', '2_1_floorplan.def')    # stamped on when here
        args = [template] if os.path.exists(template) else []
        r = subprocess.run([sys.executable, os.path.join(here, 'quickstart.py'), name, '0.6'] + args,
                           capture_output=True, text=True, timeout=300)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertIn("'legal': True", r.stdout)
        if args:
            self.assertIn('orfs_handoff.py', r.stdout)
        for f in ('floorplan', 'placed'):
            self.assertTrue(os.path.exists(os.path.join(here, 'orfs_asap7_2_1_aes_%s.png' % f)))


if __name__ == '__main__':
    unittest.main()
