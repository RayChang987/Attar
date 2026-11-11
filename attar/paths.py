"""Where the zoo of distilled designs lives. No environment variables: a design is self-contained (its IR holds
everything the fast loop reads), and a raw file is handed over explicitly where it is used (Design.export takes
the DEF to stamp onto).

The zoo is <repo>/zoo in a source checkout (when that folder exists), else ~/.cache/attar/zoo. Every function that
reads or writes the zoo takes `root=` to use another folder.
"""
import os

# The zoo is split by what the user solves: before_fp, a floorplan problem (the die / utilization, IO pins and macros
# are the answer; the result goes to a flow that draws the power grid), and after_fp, a placement problem (die, IO
# pins, macros and the power grid given; the cells are the answer). Zoo names are <mode>/<suite>/<design>.
MODES = ('before_fp', 'after_fp')

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKOUT = os.path.exists(os.path.join(REPO, 'pyproject.toml'))      # running from a source checkout, not site-packages


def zoo_dir(root=None):
    """The zoo folder: `root` when given, else <repo>/zoo in a checkout, else ~/.cache/attar/zoo."""
    if root:
        return os.path.abspath(os.path.expanduser(root))
    if CHECKOUT and os.path.isdir(os.path.join(REPO, 'zoo')):
        return os.path.join(REPO, 'zoo')
    return os.path.join(os.path.expanduser('~'), '.cache', 'attar', 'zoo')
