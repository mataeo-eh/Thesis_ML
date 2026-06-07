"""Smoke test: confirms the package imports and exposes a version.

Acts as a placeholder so the test suite is runnable from day one. Depends on the
`thesis_ml` package (src/thesis_ml/__init__.py).
"""

import thesis_ml


def test_package_has_version():
    """The package should expose a non-empty __version__ string."""
    assert isinstance(thesis_ml.__version__, str)
    assert thesis_ml.__version__
