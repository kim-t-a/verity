"""``config.py`` must describe the system that exists, not the one that did.

The module says of itself that it "deliberately holds only what the code in the
repository actually uses today". That was true when written and quietly stopped
being true as the challenge-era code was deleted around it, which is the usual
way dead configuration accumulates: nothing breaks, so nothing notices.

Dead constants are not harmless here. A hard-coded embedding dimension sitting
in the central config file is an invitation for the next feature to read it and
assume the dimension is fixed -- an assumption the design spec rules out
explicitly, and one that would then be implemented without anyone re-deciding
it.

So the property is asserted rather than trusted: every module-level constant
``config`` defines has a consumer somewhere in the backend.

The search is a whole-word text match, so a name mentioned in a *comment or
docstring* counts as a consumer even though nothing reads it. That is a real
blind spot, and this file stays clear of it by never spelling out a constant
name it might be talking about -- including, as it happens, the one above.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
CONFIG_PATH = BACKEND / "verity" / "engine" / "config.py"


def module_level_constant_names(source: str) -> set[str]:
    """The uppercase names ``config.py`` assigns at module level."""
    names: set[str] = set()
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        for target in targets:
            if isinstance(target, ast.Name) and target.id.isupper():
                names.add(target.id)
    return names


def backend_sources() -> dict[str, str]:
    """Every backend Python file except ``config.py``, read once.

    Read once rather than per name: the tree is on a 9p mount where metadata
    is slow, and searching it twenty times made this test take a minute.
    """
    sources = {}
    for root in (BACKEND / "verity", BACKEND / "tests"):
        for path in sorted(root.rglob("*.py")):
            if path != CONFIG_PATH:
                sources[path.relative_to(BACKEND).as_posix()] = path.read_text(
                    encoding="utf-8"
                )
    return sources


def test_every_constant_in_config_has_a_consumer() -> None:
    source = CONFIG_PATH.read_text(encoding="utf-8")
    sources = backend_sources()

    orphans = [
        name
        for name in sorted(module_level_constant_names(source))
        if not any(
            re.search(rf"\b{re.escape(name)}\b", text) for text in sources.values()
        )
    ]

    assert orphans == [], (
        "config.py defines constants nothing reads: "
        + ", ".join(orphans)
        + ". Either wire them up in the task that needs them, or delete them: a "
        "constant in the central config file reads as a decision that was made."
    )
