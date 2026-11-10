"""Where the zoo of distilled designs lives. No environment variables: a design is self-contained (its IR holds
everything the fast loop reads), and a raw file is handed over explicitly where it is used (Design.export takes
the DEF to stamp onto).

The zoo is <repo>/zoo in a source checkout (when that folder exists), else ~/.cache/attar/zoo. Every function that
reads or writes the zoo takes `root=` to use another folder.
"""
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKOUT = os.path.exists(os.path.join(REPO, 'pyproject.toml'))      # running from a source checkout, not site-packages


def zoo_dir(root=None):
    """The zoo folder: `root` when given, else <repo>/zoo in a checkout, else ~/.cache/attar/zoo."""
    if root:
        return os.path.abspath(os.path.expanduser(root))
    if CHECKOUT and os.path.isdir(os.path.join(REPO, 'zoo')):
        return os.path.join(REPO, 'zoo')
    return os.path.join(os.path.expanduser('~'), '.cache', 'attar', 'zoo')
