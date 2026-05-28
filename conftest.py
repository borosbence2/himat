"""Make `himat` importable in tests without an editable install.

Lets `pytest` run straight from a fresh checkout (CI, or a box that only has
torch + pytest) by putting src/ on sys.path.
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
