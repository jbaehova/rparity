"""Packaging guards for a numerical backend that must ship with the wheel."""
import json
from pathlib import Path

import pytest

import rparity._openblas as backend


@pytest.fixture
def isolated_loader():
    backend.load_library.cache_clear()
    yield
    backend.load_library.cache_clear()


def test_installed_package_does_not_search_for_an_external_backend(
    monkeypatch, tmp_path, isolated_loader
):
    package = tmp_path / "site-packages" / "rparity"
    package.mkdir(parents=True)
    monkeypatch.setattr(backend, "__file__", str(package / "_openblas.py"))

    def forbid_external_metadata(*args):
        raise AssertionError("Installed wheels must not depend on an external provider.")

    monkeypatch.setattr(backend.importlib.metadata, "version", forbid_external_metadata)
    with pytest.raises(ImportError, match="wheel is missing its numerical backend"):
        backend.load_library()


@pytest.mark.parametrize("library", ["../external.so", "/external.so"])
def test_manifest_rejects_nonrelative_library_paths(tmp_path, library):
    manifest = {"provider_version": "0.3.34.237.0", "library": library}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="package-relative path"):
        backend._bundled_library(tmp_path)


def test_manifest_rejects_an_unsupported_backend_version(tmp_path):
    manifest = {"provider_version": "0.0.0", "library": "lib/backend.so"}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="unsupported version"):
        backend._bundled_library(tmp_path)


def test_manifest_reports_a_missing_bundled_library(tmp_path):
    manifest = {"provider_version": "0.3.34.237.0", "library": "lib/backend.so"}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="numerical library is missing"):
        backend._bundled_library(tmp_path)


def test_loader_uses_the_relocated_package_bundle(monkeypatch, tmp_path, isolated_loader):
    package = tmp_path / "site-packages" / "rparity"
    bundled = package / "_openblas_libs"
    library = bundled / "lib" / "backend.so"
    library.parent.mkdir(parents=True)
    library.touch()
    manifest = {
        "provider_version": "0.3.34.237.0",
        "library": "lib/backend.so",
        "libraries": [{"path": "lib/backend.so"}],
    }
    (bundled / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(backend, "__file__", str(package / "_openblas.py"))
    loaded = []
    monkeypatch.setattr(backend.ctypes, "CDLL", lambda path: loaded.append(Path(path)))
    backend.load_library()
    assert loaded == [library]
