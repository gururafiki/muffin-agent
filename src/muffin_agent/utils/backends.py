"""Reusable backend factories for skills-enabled agents."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from deepagents.backends import CompositeBackend, FilesystemBackend

from ..sandbox import get_backend

_SKILLS_ROOT = Path(__file__).resolve().parents[1] / "skills"


def get_skills_backend(runtime: Any) -> CompositeBackend:
    """Create a CompositeBackend routing skill reads to local filesystem.

    Code execution goes to the OpenSandbox backend (default).
    Skill file reads (``/skills/``) go to the local filesystem.
    """
    sandbox = get_backend(runtime)
    skills_fs = FilesystemBackend(root_dir=_SKILLS_ROOT, virtual_mode=True)
    return CompositeBackend(
        default=sandbox,
        routes={"/skills/": skills_fs},
    )
