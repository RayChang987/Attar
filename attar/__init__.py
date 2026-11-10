"""Attar: a fast loop for physical design. Designs from any source become GPU tensors (Design), metrics evaluate
them (Evaluator), and a basic set of tensorized engines (attar.naive) makes them move. What a design can do follows
from the facts it has. See README.md."""
from .design import Design, load, zoo
from .distill import distill
from .evaluator import Evaluator
from .hub import fetch

__all__ = ['Design', 'load', 'zoo', 'fetch', 'distill', 'Evaluator']
