"""Existing frozen-dataset preparation files, preserved under train/."""
import sys
from pathlib import Path

# Preserve the original scripts' imports and bytes (their hashes are frozen).
_directory = str(Path(__file__).resolve().parent)
if _directory not in sys.path:
    sys.path.insert(0, _directory)
