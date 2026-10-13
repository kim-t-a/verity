"""Central configuration for the MC3 RAG system.

Every tunable constant lives here rather than inline, so behaviour can be
reviewed in one place. Anything that would otherwise be an unexplained number
in the middle of a function belongs in this module.

This file grows as later stages need it; it deliberately holds only what the
code in the repository actually uses today.
"""

from __future__ import annotations

# --- Ingestion --------------------------------------------------------------

#: How many bytes of each file the walker reads. That single read does two
#: jobs: it proves the file is genuinely readable (the only reliable test --
#: permission checks can lie under root and on mounted Windows drives), and it
#: captures the leading bytes used to identify the file's true format.
#: 4 KiB is far more than any format signature needs, and small enough that
#: probing even a large corpus costs almost nothing.
PROBE_BYTES = 4096
