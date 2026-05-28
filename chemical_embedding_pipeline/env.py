from __future__ import annotations

import os
from pathlib import Path


def load_dotenv_key(key: str, path: Path = Path(".env")) -> None:
    if os.environ.get(key) or not path.exists():
        return
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            if key == "OPENAI_API_KEY" and stripped.startswith("sk-"):
                os.environ[key] = stripped
                return
            continue
        name, value = stripped.split("=", 1)
        name = name.removeprefix("export").strip()
        if name != key:
            continue
        value = value.strip().strip("'").strip('"')
        if value:
            os.environ[key] = value
        return
