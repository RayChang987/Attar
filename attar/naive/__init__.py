"""Naive: the basic set of engines that makes a Design move (spec 5.1: what a design can do follows from its facts).

Each engine requires facts, works on the tensors, and writes its answer back into the Design.
  floorplan      die for a utilization (make_die), then IO pins and macros (attar/naive/flow.py)
  place          global_place + legalize (attar/naive/flow.py)
  make_die       a die and rows for a utilization (attar/naive/floorplan.py)
  place_io       IO pins on the die boundary (attar/naive/floorplan.py)
  place_macros   mixed-size GP + macro legalization on the pin-track grid: sweep line or parallel push
                 (attar/naive/floorplan.py)
  global_place   ePlace-style electrostatic placement (attar/naive/gp.py)
  legalize       multi-row-height and rail aware, tensorized (attar/naive/lg.py)
Detailed placement is not part of Naive yet.
"""
from .floorplan import make_die, place_io, place_macros
from .flow import floorplan, place
from .gp import global_place
from .lg import legalize

__all__ = ['floorplan', 'make_die', 'place', 'place_io', 'place_macros', 'global_place', 'legalize']
