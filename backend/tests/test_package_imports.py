"""The package must be importable by name, from anywhere.

This is the opposite of the test it replaces. The old suite put
``starter/app`` on ``sys.path`` so that modules could import each other by
bare name, reproducing how the container flattened the tree. That hack is
gone: ``verity`` is an installed package, so an import either resolves
through the package or fails loudly.
"""

from __future__ import annotations

import verity
import verity.engine


def test_package_imports() -> None:
    assert verity.engine is not None


def test_version_is_declared() -> None:
    assert isinstance(verity.__version__, str)
    assert verity.__version__.count(".") == 2, "expected a semantic version"
