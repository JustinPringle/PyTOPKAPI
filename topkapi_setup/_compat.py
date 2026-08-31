"""Third-party compatibility shims.

pysheds 0.5 calls ``numpy.in1d``, which NumPy removed in 2.0 (superseded by
``numpy.isin``). Until the environment pins a compatible pair (pysheds > 0.5 or
numpy < 2), restore the alias so the terrain stage runs on NumPy 2.

Import this module *before* importing pysheds. The shim is a no-op on NumPy
versions that still provide ``in1d``.
"""

import numpy as np

if not hasattr(np, "in1d"):
    # numpy.isin is the drop-in replacement; signatures match for pysheds' use.
    np.in1d = np.isin  # type: ignore[attr-defined]
