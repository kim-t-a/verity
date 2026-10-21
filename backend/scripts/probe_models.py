"""List the model ids the NVIDIA endpoint offers for this account.

A diagnostic, not part of the product: it answers "what may I set
NVIDIA_LLM_MODEL to?" without guessing from a catalogue page. It reads the
key from the repo-root .env, never prints the key, and prints only model ids.

Run with:  python -m scripts.probe_models
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from dotenv import dotenv_values

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    values = dotenv_values(REPO_ROOT / ".env")
    api_key = values.get("NVIDIA_API_KEY", "")
    base_url = (values.get("NVIDIA_API_BASE_URL", "") or "").rstrip("/")

    print(
        "key present:", bool(api_key), "| base url host:", base_url.split("//")[-1]
    )
    if not api_key or not base_url:
        print("Set NVIDIA_API_KEY and NVIDIA_API_BASE_URL in .env first.")
        return

    request = urllib.request.Request(
        f"{base_url}/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))

    for model in payload.get("data", []):
        print(model.get("id"))


if __name__ == "__main__":
    main()
