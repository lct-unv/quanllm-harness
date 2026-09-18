"""Backend availability helpers.

``importlib.util.find_spec`` only tells whether a module can be *found*, not
whether it can be *imported*: on Windows a package whose native DLL fails to
load is findable but ``import`` raises ``ImportError``. Every check that gates
behaviour on a default quantum backend must use :func:`can_import`, which
performs a real import and treats any import-time exception as unavailable.
"""

from __future__ import annotations

import importlib


def can_import(module: str) -> bool:
    """Return True only if ``module`` can actually be imported.

    Catches any exception raised while importing (including ``ImportError``
    from a failing native DLL at import time), not just ModuleNotFoundError.
    """
    try:
        importlib.import_module(module)
    except Exception:
        return False
    return True


__all__ = ["can_import"]
