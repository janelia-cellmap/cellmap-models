"""A cache of mia-train models' exports: each model converted once
(``export_mia_model``), and found again by whoever serves it next.

Converting a model takes minutes and writes the size of its weights (a
ViT-L's is over a GB), so a program that serves these models keeps the
exports in a folder of its own, ``cache``, which every function here is
given. ``cached_export`` answers a model's export, converting it when the
cache has none; ``find_export`` only looks, without the network or torch.

In ``cache`` an export is kept by its model, by the converter that made it,
and by the kind of device it was traced on (``export_dir``)::

    <org>--<repo>/<commit>/<converter>/<cuda|cpu>
    local/<folder name>-<fingerprint>/<converter>/<cuda|cpu>

- The converter is a fingerprint of ``mia_model``'s source
  (``converter_version``): a cellmap-models whose converter changed converts
  again rather than serving an export the old one made.
- The device kind is the one of the process that converts: on CUDA the
  network is traced in bf16, else on the CPU in fp32, so a server serves the
  export traced on its own kind of device.

The conversion runs in a Python process of its own (``export_in_subprocess``):
mia-train's top-level packages (``models``, ``layers``, ``distributed``...)
stay imported once imported, and its ``distributed`` breaks Dask's in the
same process.

Two processes converting the same model at once each convert into a folder
of their own beside the export's and rename it into place: the first rename
wins and the other copy is dropped, so nobody reads a half-written export.
A conversion killed outright (SIGKILL) leaves its folder behind, which the
next conversion removes once it has not changed for an hour
(``remove_stale_work``). A conversion done also removes the same model's
exports for the same device that other converters made, once none has been
opened for UNUSED_EXPORT_DAYS (``remove_other_converters``): those of an
install nobody runs any more only take room, while two installs with
different converters that share a cache each keep their own.
"""

import functools
import hashlib
import importlib
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# The files of a finished export (export_mia_model's).
EXPORT_FILES = ("metadata.json", "model.ts")
# The kinds of device an export is traced on, the one find_export prefers first.
DEVICE_KINDS = ("cuda", "cpu")


def _mia_model():
    """``mia_model``, as sys.modules has it (imported now when it is not)."""
    return importlib.import_module("cellmap_models.model_export.mia_model")


def device_kind() -> str:
    """The kind of device an export made here is traced on: "cuda" where a
    CUDA GPU is (traced in bf16 there), else "cpu" (in fp32; an Apple GPU
    serves that one)."""
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def converter_version() -> str:
    """A fingerprint of the converter installed: the first 12 hex digits of
    the SHA-1 of ``mia_model``'s source file, or "unknown" when it has none.
    An export is cached under it, so a cellmap-models whose converter changed
    (what an export declares, how it wraps the network) converts again
    rather than serving an export the old one made; the package's version
    number stays the same across its commits, so it cannot say this.

    The file is read again whenever its time or size changed
    (``_source_fingerprint``), so a process left running across an update of
    cellmap-models looks for the exports the new one makes."""
    source = getattr(_mia_model(), "__file__", None)
    try:
        stat = os.stat(source)
        return _source_fingerprint(source, stat.st_mtime_ns, stat.st_size)
    except (OSError, TypeError):  # TypeError: no file
        return "unknown"


@functools.lru_cache(maxsize=8)
def _source_fingerprint(source: str, mtime_ns: int, size: int) -> str:
    """The first 12 hex digits of the SHA-1 of the file ``source``, read once
    for each time and size it has had."""
    with open(source, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()[:12]


def model_dir(cache, repo: Optional[str] = None, commit: Optional[str] = None, path: Optional[str] = None) -> Path:
    """The folder in ``cache`` of the model of ``repo`` at ``commit`` (a full
    commit SHA), or of the folder ``path``, as the converter installed makes
    it (``converter_version``), which holds its exports by device kind:
    ``<org>--<repo>/<commit>/<converter>``, or ``local/<folder
    name>-<fingerprint>/<converter>`` with the fingerprint of the folder's
    path and its MIA_FILES' sizes and times, so that new weights in the same
    folder are converted again."""
    if path is None:
        return Path(cache) / repo.replace("/", "--") / commit / converter_version()
    fingerprint = hashlib.sha1(os.path.abspath(path).encode())
    for name in _mia_model().MIA_FILES:
        stat = os.stat(os.path.join(path, name))
        fingerprint.update(f"|{name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    folder = f"{os.path.basename(os.path.abspath(path))}-{fingerprint.hexdigest()[:12]}"
    return Path(cache) / "local" / folder / converter_version()


def export_dir(cache, repo: Optional[str] = None, commit: Optional[str] = None, path: Optional[str] = None,
               device: Optional[str] = None) -> Path:
    """The folder in ``cache`` of the export of the model (``model_dir``)
    traced on ``device``, a kind of DEVICE_KINDS (this machine's,
    ``device_kind``, by default)."""
    return model_dir(cache, repo, commit, path) / (device or device_kind())


def is_export(folder) -> bool:
    """Whether ``folder`` holds a finished export (its EXPORT_FILES)."""
    return all(os.path.isfile(os.path.join(folder, name)) for name in EXPORT_FILES)


# What the conversion's own Python runs: the export, and nothing of the
# program that asked for it, so that no package of the names mia-train's are
# is imported before it.
_EXPORT_PROGRAM = (
    "import sys\n"
    "from cellmap_models.model_export.mia_model import export_mia_model\n"
    "export_mia_model(sys.argv[1], sys.argv[2])\n"
)


def export_in_subprocess(source, out) -> None:
    """``export_mia_model(source, out)`` in a Python process of its own, this
    process's python (``-P``: not the working directory's modules). Its log
    is this process's; a failure is a RuntimeError pointing there.

    Never in a process that serves: mia-train's top-level packages stay
    imported once its model is loaded, and its ``distributed`` breaks Dask's
    there; ``mia_model``'s loader refuses besides when a package named
    ``models``, ``layers``, ``utils`` or ``distributed`` is imported already.
    """
    result = subprocess.run([sys.executable, "-P", "-c", _EXPORT_PROGRAM, str(source), str(out)])
    if result.returncode != 0:
        raise RuntimeError(
            f"Converting the mia-train model {source} into a cellmap-models export failed (exit status "
            f"{result.returncode}); its traceback is just above in this log"
        )


# What a conversion's work folder is named, beside its export's: ``.<device
# kind>.converting-<random>`` (convert_once).
WORK_SUFFIX = ".converting-"
# How long a work folder may go unchanged before it is taken to be a killed
# conversion's: a conversion under way writes into it, and the longest it
# goes without (loading a ViT-L, tracing it on a CPU) is minutes.
STALE_WORK_SECONDS = 3600


def _last_change(folder) -> float:
    """The newest modification time of ``folder`` or anything in it."""
    newest = os.stat(folder).st_mtime
    for root, dirs, files in os.walk(folder):
        for name in dirs + files:
            newest = max(newest, os.lstat(os.path.join(root, name)).st_mtime)
    return newest


def remove_stale_work(root) -> None:
    """Remove the work folders under ``root`` (the cache) of conversions
    killed before they could remove their own: a SIGKILL ends a process
    without running convert_once's ``finally``, and leaves a folder the size
    of the model's weights behind. One unchanged for STALE_WORK_SECONDS is
    such a folder; one changed since is a conversion's under way."""
    now = time.time()
    for work in list(Path(root).glob(f"**/.*{WORK_SUFFIX}*")):
        try:
            stale = now - _last_change(work) > STALE_WORK_SECONDS
        except OSError:  # removed meanwhile, by its own conversion or another sweep
            continue
        if stale:
            logger.info(f"Removing {work}, the work folder of a conversion that was killed")
            shutil.rmtree(work, ignore_errors=True)


# How long another converter's export may go unopened before a conversion
# removes it (remove_other_converters): long enough that an install in use,
# which opens its exports (mark_used), never loses one.
UNUSED_EXPORT_DAYS = 14
# An export opened is marked used (its folder's time set to now) when its
# mark is older than this, not at every opening.
MARK_USED_SECONDS = 24 * 3600


def mark_used(folder) -> None:
    """Note that the export in ``folder`` was opened now, by setting the
    folder's modification time, which remove_other_converters reads; at most
    once every MARK_USED_SECONDS. A folder that cannot be marked (not this
    user's, a read-only cache) is left as it is."""
    try:
        if time.time() - os.stat(folder).st_mtime > MARK_USED_SECONDS:
            os.utime(folder)
    except OSError:
        pass


def _remove_folder(folder) -> None:
    """Remove ``folder`` from the cache: renamed first to a work folder's
    name beside it, which is atomic, so that no process finds half of it
    under its own name, then deleted. Nothing is raised: what could not be
    renamed stays, and what could not be deleted is a work folder, which
    remove_stale_work removes later."""
    folder = Path(folder)
    aside = folder.parent / f".{folder.name}{WORK_SUFFIX}removed-{os.getpid()}-{time.time_ns()}"
    try:
        os.rename(folder, aside)
    except OSError:
        return
    shutil.rmtree(aside, ignore_errors=True)


def remove_other_converters(target) -> None:
    """Remove the exports for ``target``'s device of the same model made by
    other converters than ``target``'s, once they have not been opened
    (``mark_used``) for UNUSED_EXPORT_DAYS: ``<converter>/<device>`` beside
    ``target``'s ``<converter>`` in the model's folder (``model_dir``).

    One opened since is another install's in use (a checkout or deployment
    with another cellmap-models that shares this cache): removing it would
    make that install convert again, and remove this one's in turn, and
    could take the files from under a process that is loading them."""
    target = Path(target)
    unused_since = time.time() - UNUSED_EXPORT_DAYS * 24 * 3600
    try:
        converters = list(target.parent.parent.iterdir())
    except OSError:
        return
    for converter in converters:
        old = converter / target.name
        if converter == target.parent:
            continue
        try:
            if not is_export(old) or os.stat(old).st_mtime > unused_since:
                continue
        except OSError:  # removed meanwhile
            continue
        logger.info(f"Removing {old}, an export of the same model by another converter, "
                    f"not opened for {UNUSED_EXPORT_DAYS} days")
        _remove_folder(old)
        try:
            converter.rmdir()  # when that was its last export
        except OSError:
            pass


def convert_once(source, target, cache=None) -> None:
    """Export the mia-train model at ``source`` (a folder with its MIA_FILES)
    into ``target``, unless another process does first, in a process of its
    own (``export_in_subprocess``).

    It is exported into a new folder beside ``target`` and renamed into
    place, which is atomic: a process that finds ``target`` finds the whole
    export. When another rename got there first, that export is kept and
    this one dropped; a ``target`` that is no whole export (what a removal
    cut short left of one) is replaced. Before, ``cache`` (``target``'s
    folder when none is given) is rid of killed conversions' work folders
    (``remove_stale_work``); after, of the same model's exports by other
    converters that are no longer used (``remove_other_converters``).
    """
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    remove_stale_work(cache if cache is not None else target.parent)
    work = tempfile.mkdtemp(dir=target.parent, prefix=f".{target.name}{WORK_SUFFIX}")
    try:
        # A folder of its own inside work, which the exporter creates.
        out = os.path.join(work, "export")
        logger.info(f"Converting the mia-train model {source} into a cellmap-models export in {target} (once)")
        export_in_subprocess(source, out)
        if not is_export(out):
            raise RuntimeError(f"export_mia_model wrote no {' and '.join(EXPORT_FILES)} into {out}")
        try:
            os.rename(out, target)
        except OSError:
            if is_export(target):
                logger.info(f"{target} was converted by another process meanwhile; serving that one")
            elif os.path.lexists(target):
                # Part of an export, which no conversion leaves (each renames a whole one into place).
                logger.info(f"Replacing {target}, which is not a whole export")
                _remove_folder(target)
                os.rename(out, target)
            else:
                raise
    finally:
        shutil.rmtree(work, ignore_errors=True)
    remove_other_converters(target)


def cached_export(cache, repo: Optional[str] = None, revision: Optional[str] = None,
                  path: Optional[str] = None) -> str:
    """The folder in ``cache`` of the export of a mia-train model, a Hugging
    Face ``repo`` (at ``revision``) or a local folder ``path``, converted now
    when it is not cached yet, and marked used.

    A repo's snapshot is downloaded first (its MIA_FILES and CARD_FILES
    only), which also says the commit its export is cached under. With the
    Hub out of reach, Hugging Face answers the snapshot its own cache has for
    ``revision``, whatever files that holds: enough to find an export made
    already, but not to convert one unless the weights are among them, which
    is refused, saying so.

    Raises:
        RuntimeError: the model must be converted and its files are not all
            there (the weights could not be downloaded), or the conversion
            failed.
    """
    mia_model = _mia_model()
    if path:
        source, target = path, export_dir(cache, path=path)
    else:
        from huggingface_hub import snapshot_download

        files = [*mia_model.MIA_FILES, *getattr(mia_model, "CARD_FILES", ())]
        source = snapshot_download(repo, revision=revision, allow_patterns=files)
        # The snapshot's folder is its commit's SHA.
        target = export_dir(cache, repo, os.path.basename(os.path.normpath(source)))
    if not is_export(target):
        missing = [name for name in mia_model.MIA_FILES if not os.path.isfile(os.path.join(source, name))]
        if missing:
            raise RuntimeError(
                f"The mia-train model {repo} cannot be converted: {', '.join(missing)} could not be downloaded "
                f"from Hugging Face (only part of the repository is in the local cache, {source}). Is this "
                "machine offline, or the Hub out of reach? Try again once it can be reached."
                if repo else
                f"The mia-train model in {path} cannot be converted: it has no {', '.join(missing)}"
            )
        convert_once(source, target, cache)
    mark_used(target)
    return str(target)


def find_export(cache, repo: Optional[str] = None, revision: Optional[str] = None,
                path: Optional[str] = None) -> Optional[Path]:
    """The folder in ``cache`` of the export of the model, as far as is known
    without the network or torch, and without converting: a repo's commit is
    the one Hugging Face's local cache has for ``revision``; of the exports
    of each device kind, the first DEVICE_KINDS has (their metadata says the
    same). None when there is none.

    Raises:
        OSError: the folder ``path``'s files cannot be read.
    """
    if path:
        models = model_dir(cache, path=path)
    else:
        from huggingface_hub import try_to_load_from_cache

        found = try_to_load_from_cache(repo, "config.json", revision=revision)
        if not isinstance(found, str):
            return None
        models = model_dir(cache, repo, Path(found).parent.name)
    return next((models / kind for kind in DEVICE_KINDS if is_export(models / kind)), None)
