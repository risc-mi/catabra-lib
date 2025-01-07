#  Copyright (c) 2025. RISC Software GmbH.
#  All rights reserved.

try:
    from . import _matplotlib as mpl_backend
except ImportError:
    mpl_backend = None

try:
    from . import _plotly as plotly_backend
except ImportError:
    plotly_backend = None


__all__ = ["mpl_backend", "plotly_backend"]
