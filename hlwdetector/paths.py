"""Repo-root anchoring for portable artifact paths.

All paths serialized to disk (config.json, model.json, etc.) are stored relative
to the repository root via ``to_repo_rel`` and resolved back to absolute paths on
the current machine via ``resolve``. This keeps experiment artifacts portable
across environments (e.g. a local laptop and the PACE HPC) and users, as long as
data and weights live inside the repo.
"""

from __future__ import annotations

from pathlib import Path

# hlwdetector/paths.py -> repo root is one level up from the package dir.
REPO_ROOT = Path(__file__).resolve().parents[1]


def to_repo_rel(path: "str | Path") -> str:
    """Serialize an absolute path to a POSIX string relative to the repo root.

    Paths outside the repo are returned absolute (rare; e.g. an external weights
    cache). POSIX separators keep artifacts portable across Windows/WSL/Linux.
    """
    p = Path(path).resolve()
    try:
        return p.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(p)


def resolve(path: "str | Path") -> Path:
    """Resolve a (possibly repo-relative) stored path to an absolute Path.

    An already-absolute path is returned as-is, so artifacts written before this
    change — or paths pointing outside the repo — keep working where the files
    exist on this machine.
    """
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()
