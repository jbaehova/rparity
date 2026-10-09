"""Load rparity's private, vendored LP64 numerical library."""
from __future__ import annotations

import ctypes
import importlib.metadata
import json
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

_PROVIDER_VERSION = "0.3.34.237.0"
_DLL_DIRECTORY_HANDLES: list[Any] = []


def _source_root(package_directory: Path) -> Path | None:
    """Restrict the development fallback to this project's src checkout."""
    if package_directory.parent.name != "src":
        return None
    root = package_directory.parent.parent
    if not (root / "pyproject.toml").is_file() or not (root / "hatch_build.py").is_file():
        return None
    return root


def _bundled_library(directory: Path) -> Path:
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["provider_version"] != _PROVIDER_VERSION:
        raise RuntimeError("The bundled rparity numerical backend has an unsupported version.")
    relative = Path(manifest["library"])
    if relative.anchor or ".." in relative.parts:
        raise RuntimeError("The bundled numerical library must use a package-relative path.")
    library = directory / relative
    if not library.is_file():
        raise RuntimeError("The bundled rparity numerical library is missing. Reinstall rparity.")
    if sys.platform == "win32":
        for parent in sorted({(directory / item["path"]).parent
                              for item in manifest["libraries"]}):
            _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(parent)))
    return library


@lru_cache(maxsize=1)
def load_library() -> ctypes.CDLL:
    """Load the wheel's library without searching the host or importing R."""
    package_directory = Path(__file__).resolve().parent
    bundled = package_directory / "_openblas_libs"
    if bundled.is_dir():
        return ctypes.CDLL(str(_bundled_library(bundled)))
    source_root = _source_root(package_directory)
    if source_root is not None:
        editable_bundle = source_root / "build" / "rparity-openblas"
        if editable_bundle.is_dir():
            return ctypes.CDLL(str(_bundled_library(editable_bundle)))
        try:
            version = importlib.metadata.version("scipy-openblas32")
            if version != _PROVIDER_VERSION:
                raise ImportError(f"Local development requires scipy-openblas32=={_PROVIDER_VERSION}.")
            from scipy_openblas32 import dll
            return dll
        except ImportError as exc:
            raise ImportError("Build the editable project or run uv sync for local development.") from exc
    raise ImportError("The rparity wheel is missing its numerical backend. Reinstall rparity.")
