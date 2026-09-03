"""Forcing builders (M3).

One array is the target: rainfall depth of shape ``(n_t, n_cells)``, on the
canonical cell order, written to ``rainfields.h5``.  Everything in this package
is machinery for filling it.

Modules land in build order:

===================  ============================================  ========
``interpolate.py``   the ``W``-builders (mean/thiessen/idw/kriging)  present
``gauges.py``        manifest + measurements IO, clock, gaps         present
``rainfields.py``    the ``(n_t, n_cells)`` writer                   present
``gridded.py``       CHIRPS/IMERG resample onto grid                 to do
===================  ============================================  ========
"""

from topkapi_setup.forcing.gauges import (  # noqa: F401
    Timeline,
    aggregate,
    align_to_clock,
    coverage,
    disaggregate,
    gauge_xy,
    read_manifest,
    read_measurements,
)
from topkapi_setup.forcing.rainfields import (  # noqa: F401
    build_and_write_rainfields,
    build_rainfield,
    check_cell_order,
    read_rainfields,
    write_rainfields,
)
from topkapi_setup.forcing.interpolate import (  # noqa: F401
    DEFAULT_BUFFER_M,
    DEFAULT_IDW_POWER,
    DEFAULT_METHOD,
    METHODS,
    build_weights,
    catchment_cell_xy,
    renormalise,
    select_gauges,
)
