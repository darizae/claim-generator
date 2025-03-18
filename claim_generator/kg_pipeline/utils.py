import re
from pathlib import Path

def sanitize_filename(text: str) -> str:
    """
    Replaces slashes and invalid filename characters with underscores.
    """
    return re.sub(r"[^a-zA-Z0-9_\-\.]", "_", text)

def ensure_dir(path: Path):
    """
    Creates a directory if it doesn't exist.
    """
    path.mkdir(parents=True, exist_ok=True)
