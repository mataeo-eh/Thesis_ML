"""Smoke test: confirms the package imports and exposes a version.

Acts as a placeholder so the test suite is runnable from day one. Depends on the
`thesis_diffusion` package (src/thesis_diffusion/__init__.py).
"""

import thesis_diffusion


def test_package_has_version():
    """The package should expose a non-empty __version__ string."""
    assert isinstance(thesis_diffusion.__version__, str)
    assert thesis_diffusion.__version__
