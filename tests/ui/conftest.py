# -*- coding: utf-8 -*-
"""Make the Auto Arena service modules importable from the repository root.

The real ``features/__init__.py`` imports every Streamlit feature, but Streamlit
is a UI-only dependency and is not installed by ``pip install -e .[dev]`` used
in CI. Register lightweight namespace packages instead, so the pure-Python
service modules can be tested without pulling in the whole UI.
"""

import sys
import types
from pathlib import Path

UI_DIR = Path(__file__).resolve().parents[2] / "ui"
if str(UI_DIR) not in sys.path:
    sys.path.insert(0, str(UI_DIR))


def _ensure_namespace(name: str, path: Path) -> None:
    """Register ``name`` as an importable package backed by ``path``."""
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module


_ensure_namespace("features", UI_DIR / "features")
_ensure_namespace("features.auto_arena", UI_DIR / "features" / "auto_arena")
_ensure_namespace("features.auto_arena.services", UI_DIR / "features" / "auto_arena" / "services")
