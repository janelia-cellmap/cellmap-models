"""The cache of mia-train exports (``model_export.mia_cache``): what a
conversion keeps and removes, and what it refuses to convert. The export
itself is faked (a folder with its two files): nothing here imports
mia-train or traces a network."""

import os
import time
from pathlib import Path

import pytest

from cellmap_models.model_export import mia_cache, mia_model

COMMIT = "1ceca0ef" + "0" * 32


def _placeholder_export(folder, days_old=0):
    """A folder with an export's files, last opened ``days_old`` days ago."""
    folder.mkdir(parents=True, exist_ok=True)
    for name in mia_cache.EXPORT_FILES:
        (folder / name).write_text("{}")
    opened = time.time() - days_old * 24 * 3600
    os.utime(folder, (opened, opened))
    return folder


def test_a_conversion_removes_killed_conversions_folders_and_only_the_unused_exports_of_other_converters(
        monkeypatch, tmp_path):
    """A conversion killed outright left its work folder, and a converter no
    install runs any more left its export for this device: converting
    removes those. It keeps the export of another install in use (two
    installs sharing a cache would otherwise each reconvert what the other
    had just removed), a conversion under way, and the other device's
    export. What a removal cut short left under the export's own name is
    replaced."""
    model = tmp_path / "ai-hhmi--lmd-vitl" / COMMIT
    killed, running = model / "new" / ".cuda.converting-1", model / "old" / ".cpu.converting-2"
    for work in (killed, running):
        (work / "export").mkdir(parents=True)
    hours_ago = time.time() - 2 * mia_cache.STALE_WORK_SECONDS
    for path in (killed / "export", killed):
        os.utime(path, (hours_ago, hours_ago))
    unused = mia_cache.UNUSED_EXPORT_DAYS + 1
    old_cuda = _placeholder_export(model / "old" / "cuda", unused)
    old_cpu = _placeholder_export(model / "old" / "cpu", unused)
    in_use = _placeholder_export(model / "other" / "cuda", days_old=3)
    half = model / "new" / "cuda"
    half.mkdir(parents=True)
    (half / "model.ts").write_text("{}")
    monkeypatch.setattr(mia_cache, "export_in_subprocess", lambda source, out: _placeholder_export(Path(out)))

    mia_cache.convert_once(tmp_path / "source", model / "new" / "cuda", tmp_path)
    assert mia_cache.is_export(model / "new" / "cuda") and not killed.exists() and running.exists()
    assert not old_cuda.exists()
    assert mia_cache.is_export(old_cpu) and mia_cache.is_export(in_use)


def test_a_model_is_converted_once_and_found_again_without_converting(monkeypatch, tmp_path):
    """A local folder's export is made on the first ask, answered from the
    cache on the second, and found by ``find_export``, which converts
    nothing; new weights in the same folder are another model."""
    source = tmp_path / "model"
    source.mkdir()
    for name in mia_model.MIA_FILES:
        (source / name).write_text("{}")
    converted = []
    monkeypatch.setattr(mia_cache, "device_kind", lambda: "cpu")
    monkeypatch.setattr(mia_cache, "export_in_subprocess",
                        lambda source, out: converted.append(source) or _placeholder_export(Path(out)))
    cache = tmp_path / "cache"

    assert mia_cache.find_export(cache, path=str(source)) is None and not converted
    first = mia_cache.cached_export(cache, path=str(source))
    assert mia_cache.cached_export(cache, path=str(source)) == first and len(converted) == 1
    assert str(mia_cache.find_export(cache, path=str(source))) == first
    (source / "model.safetensors").write_text("new weights")
    assert mia_cache.find_export(cache, path=str(source)) is None


def test_a_repo_whose_weights_are_not_cached_is_refused_by_name_when_the_hub_is_out_of_reach(monkeypatch, tmp_path):
    """Only the repo's config.json was in Hugging Face's cache; with the Hub
    out of reach it answers that snapshot, and the conversion would fail on
    it without saying why."""
    import huggingface_hub

    snapshot = tmp_path / "hub" / "snapshots" / COMMIT
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    monkeypatch.setattr(huggingface_hub, "snapshot_download",
                        lambda repo, revision=None, allow_patterns=None: str(snapshot))
    monkeypatch.setattr(mia_cache, "device_kind", lambda: "cpu")
    monkeypatch.setattr(mia_cache, "export_in_subprocess",
                        lambda source, out: pytest.fail("converted without its weights"))

    with pytest.raises(RuntimeError, match="ai-hhmi/lmd-vitl.*model.safetensors could not be downloaded.*offline"):
        mia_cache.cached_export(tmp_path / "cache", repo="ai-hhmi/lmd-vitl")


def test_an_updated_converter_is_seen_by_a_process_that_was_running(monkeypatch, tmp_path):
    """The fingerprint is read again when the converter's file changes: a
    process left running across an update of cellmap-models looks for
    exports under the new converter's."""
    source = tmp_path / "mia_model.py"
    source.write_text("VERSION = 1\n")
    monkeypatch.setattr(mia_model, "__file__", str(source))
    first = mia_cache.converter_version()
    source.write_text("VERSION = 22\n")
    assert len(first) == 12 and mia_cache.converter_version() not in (first, "unknown")
