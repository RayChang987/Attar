"""A CUDA out-of-memory error inside an Attar step is explained (attar.gpu.GPUOutOfMemory): the step, the design,
the ways out; still a torch.cuda.OutOfMemoryError; explained once through nested steps."""
import unittest
from unittest import mock

import torch

from common import SMALL, need

from attar.gpu import GPUOutOfMemory
from attar.timing.sta import Timer


class OutOfMemoryTest(unittest.TestCase):
    def test_timing_oom_is_explained(self):
        d = need(SMALL)
        boom = torch.cuda.OutOfMemoryError('CUDA out of memory. Tried to allocate 2.00 GiB')
        with mock.patch.object(Timer, 'run', side_effect=boom):
            with self.assertRaises(torch.cuda.OutOfMemoryError) as cm:   # still a torch OOM for old handlers
                d.timing()
        e = cm.exception
        self.assertIsInstance(e, GPUOutOfMemory)
        msg = str(e)
        self.assertIn('d.timing', msg)
        self.assertIn('%d cells' % len(d.names['cells']), msg)
        self.assertIn("device='cpu'", msg)
        self.assertIn('Tried to allocate 2.00 GiB', msg)
        self.assertEqual(msg.count('the GPU ran out of memory'), 1)

    def test_nested_steps_explain_once(self):
        from attar import naive
        d = need(SMALL)
        boom = torch.cuda.OutOfMemoryError('CUDA out of memory.')
        with mock.patch.object(naive.gp, 'Grid', side_effect=boom):
            with self.assertRaises(GPUOutOfMemory) as cm:
                d.place(verbose=False)                                  # d.place -> naive.global_place
        msg = str(cm.exception)
        self.assertEqual(msg.count('the GPU ran out of memory'), 1)
        self.assertIn('naive.global_place', msg)


if __name__ == '__main__':
    unittest.main()
