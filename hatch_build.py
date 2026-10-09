"""Vendor the pinned numerical backend into correctly tagged platform wheels.

scipy-openblas32 supports build-time use and vendoring, rather than external
runtime dependencies. Its repaired library paths are preserved relative to the
original package root, including adjacent support-library directories.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import shutil
import tempfile
from email.parser import Parser
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

PROVIDER = "scipy-openblas32"
PROVIDER_VERSION = "0.3.34.237.0"


def _is_library(path: Path) -> bool:
    return path.name.endswith((".dylib", ".dll", ".so")) or ".so." in path.name


def _platform_tag(distribution: importlib.metadata.Distribution) -> str:
    wheel_metadata = distribution.read_text("WHEEL")
    if wheel_metadata is None:
        raise RuntimeError("The OpenBLAS build dependency must be installed from a wheel.")
    tags = Parser().parsestr(wheel_metadata).get_all("Tag", [])
    platforms = list(dict.fromkeys(tag.rsplit("-", 1)[-1] for tag in tags))
    if not platforms or "any" in platforms:
        raise RuntimeError("The OpenBLAS provider did not declare a native platform tag.")
    return "py3-none-" + ".".join(platforms)


def _vendor(destination: Path) -> str:
    import scipy_openblas32 as provider

    distribution = importlib.metadata.distribution(PROVIDER)
    if distribution.version != PROVIDER_VERSION:
        raise RuntimeError(f"The numerical build dependency must be {PROVIDER}=={PROVIDER_VERSION}.")
    library_directory = Path(provider.get_lib_dir()).resolve()
    package_root = library_directory.parent
    installation_root = package_root.parent
    main_name = provider.get_library(fullname=True)
    main_path = library_directory / main_name
    if not _is_library(main_path):
        candidates = sorted(library_directory.glob(main_path.stem + "*.dll"))
        if len(candidates) != 1:
            raise RuntimeError("Could not identify the provider's Windows OpenBLAS library.")
        main_path = candidates[0]
    libraries = []
    for record in distribution.files or []:
        source = Path(str(distribution.locate_file(record)))
        if not _is_library(source):
            continue
        relative = source.relative_to(installation_root)
        if ".." in relative.parts:
            raise RuntimeError("The provider library layout escapes its installation root.")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        # Wheel records contain regular files. Dereferencing an installed symlink
        # preserves its advertised filename without a link to the build machine.
        shutil.copyfile(source, target)
        libraries.append({
            "path": relative.as_posix(),
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "size": target.stat().st_size,
        })
    main_relative = main_path.relative_to(installation_root).as_posix()
    if not (destination / main_relative).is_file():
        raise RuntimeError("The provider wheel did not include its main numerical library.")
    license_records = [record for record in distribution.files or []
                       if "license" in str(record).lower()]
    if not license_records:
        raise RuntimeError("The numerical provider license notices are missing.")
    for record in license_records:
        source = Path(str(distribution.locate_file(record)))
        if source.is_file():
            target = destination / "licenses" / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    shutil.copyfile(
        Path(__file__).parent / "LICENSES" / "GNU-LGPL-2.1.txt",
        destination / "licenses" / "GNU-LGPL-2.1.txt",
    )
    tag = _platform_tag(distribution)
    manifest = {
        "provider": PROVIDER,
        "provider_version": PROVIDER_VERSION,
        "wheel_tag": tag,
        "library": main_relative,
        "libraries": libraries,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return tag


def _existing_bundle_tag(destination: Path) -> str | None:
    """Reuse identical editable libraries so an active process keeps its files."""
    try:
        manifest = json.loads((destination / "manifest.json").read_text())
        distribution = importlib.metadata.distribution(PROVIDER)
        tag = _platform_tag(distribution)
        if (manifest["provider_version"] != PROVIDER_VERSION
                or distribution.version != PROVIDER_VERSION or manifest["wheel_tag"] != tag):
            return None
        for record in manifest["libraries"]:
            relative = Path(record["path"])
            if relative.anchor or ".." in relative.parts:
                return None
            digest = hashlib.sha256((destination / relative).read_bytes()).hexdigest()
            if digest != record["sha256"]:
                return None
        if not manifest["libraries"]:
            return None
        return tag
    except (OSError, ValueError, KeyError):
        return None


class CustomBuildHook(BuildHookInterface):
    """Keep backend vendoring inside the wheel build, including editable builds."""

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        if version == "editable":
            destination = Path(self.root) / "build" / "rparity-openblas"
            tag = _existing_bundle_tag(destination)
            if tag is None:
                if destination.exists():
                    shutil.rmtree(destination)
                destination.mkdir(parents=True)
                tag = _vendor(destination)
            else:
                shutil.copyfile(
                    Path(self.root) / "LICENSES" / "GNU-LGPL-2.1.txt",
                    destination / "licenses" / "GNU-LGPL-2.1.txt",
                )
        else:
            self._staging = tempfile.TemporaryDirectory(prefix="rparity-openblas-")
            destination = Path(self._staging.name)
            tag = _vendor(destination)
        build_data["tag"] = tag
        build_data["pure_python"] = False
        if version != "editable":
            build_data["force_include"][str(destination)] = "rparity/_openblas_libs"

    def finalize(self, version: str, build_data: dict[str, Any], artifact_path: str) -> None:
        if version != "editable":
            self._staging.cleanup()
