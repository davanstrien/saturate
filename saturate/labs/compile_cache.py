"""Lab helpers: not part of the contract; may change or be removed in any release, without
deprecation.

`compile_cache` keeps a serving engine's start-up caches (compiled graphs, Triton and FlashInfer
kernels, autotune results) as one archive in a Bucket you own, so the next Job on the same
engine, model and GPU can reuse them. Put it around the code that starts the engine:

    from saturate import Engine, pump
    from saturate.labs.compile_cache import compile_cache, default_key

    key = default_key("vllm", MODEL, image=IMAGE)
    with compile_cache("hf://buckets/<you>/jobs-cache", engine="vllm", key=key) as cache:
        with Engine(MODEL, engine="vllm") as endpoint:
            cache.save()  # keep the boot's kernels even if the rest of the run fails
            pump(...)
    print(cache.report.to_json())

Without a with block: `cache.restore()` before the server starts, `cache.save()` when you
want, `cache.close()` at the end. Any server works; `Engine` is not required.

`restore()` sets the profile's environment variables (only those you have not set), downloads
`<location>/<key>.tar` if it exists and unpacks it into the engine's cache folders. A file that
already exists is kept. `save()` lists the cache folders again (path and size of each file) and
uploads a new archive of all of them only if a file is new or changed size since the last
restore or save. Whether the engine reuses a restored file is the engine's decision: it keys
its own cache folders by hashes this helper does not compute. A key that is too coarse costs a
recompile and a larger archive, not wrong output.

Failures do not stop the run: a failed download or upload logs a warning and is recorded in
`cache.report`. After a failed restore of an archive that exists, nothing is saved, so a good
archive is never replaced by a partial one. One writer per key at a time: when two Jobs save
the same key, the last one wins and the other's new files are lost (a warning says so).

The archive holds code that the engine loads and runs (.py, .so, cubins). Keep it in your own
private bucket (a missing bucket is created private), and never restore an archive that other
people can write to.
"""

from __future__ import annotations

import dataclasses
import fnmatch
import getpass
import importlib.metadata
import json
import logging
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field

import fsspec

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Profile:
    """What compile_cache archives for one engine.

    groups: named sets of cache folders; `compile_cache(skip=...)` leaves groups out. A folder
        starts with `~` or `/`; `{user}` is the user name, `{tmp}` the temporary folder (TMPDIR,
        as torch uses it), `{vllm}` and `{sglang}` the engines' cache roots (VLLM_CACHE_ROOT,
        SGLANG_CACHE_DIR, or their defaults). `$NAME` is the folder in environment variable
        NAME, when it is set. Other variables that move a cache (FLASHINFER_WORKSPACE_BASE,
        CUDA_CACHE_PATH, ...) are not followed: list the folder in a Profile of your own.
    never: files never archived or restored: a pattern matched against every part of a file's
        path inside its folder (`*.lock`, `runs`).
    env: environment variables set before the engine starts, unless already set.
    """

    groups: dict[str, tuple[str, ...]]
    never: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)


# Never in any archive: locks, temporary files, half-written staging folders.
_COMMON_NEVER = ("*.lock", "*.tmp", "tmp.*", ".staging-*", ".cc-*")

# Not archived: the CUDA driver's PTX JIT cache (~/.nv/ComputeCache; {sglang}/nv). Measured on
# rtx-pro-6000 (2026-11-19), restoring it did not change server-ready time on vLLM or SGLang,
# while it was 60-80% of each archive and grew by ~90 MB on every warm run. Add it in a
# Profile of your own if a measurement says otherwise.
VLLM = Profile(
    # Subfolders of the vLLM cache root, named one by one: the root also holds downloaded
    # assets and, when configured there, media and LoRA caches.
    groups={
        "compile": (
            "{vllm}/torch_compile_cache",
            "{vllm}/torch_aot_compile",
            "{vllm}/startup_plan",
            "{vllm}/modelinfos",
            "{tmp}/torchinductor_{user}",
        ),
        "triton": ("~/.triton", "$TRITON_CACHE_DIR"),
        "flashinfer": ("~/.cache/flashinfer", "{vllm}/flashinfer_autotune_cache"),
        "deep_gemm": ("{vllm}/deep_gemm",),
    },
    never=_COMMON_NEVER,
    # torch's first inductor autotuner otherwise points TRITON_CACHE_DIR at the inductor cache
    # of that moment, which moves between a cold and a warm run: the warm run then misses.
    env={"TRITON_CACHE_DIR": "~/.triton/cache"},
)

SGLANG = Profile(
    # Subfolders of the SGLang cache root, named one by one: the root also holds files that
    # must never move to another machine (kernels.lock from the image, gpu_p2p_* per host,
    # model_streamer, ple, paged_experts). The `~/...` folders are where older images write.
    groups={
        "compile": ("{sglang}/inductor", "{sglang}/torch_compile_cache", "{tmp}/torchinductor_{user}"),
        "jit": ("{sglang}/jit", "{sglang}/cute_aot", "{sglang}/tilelang", "{sglang}/humming/cache"),
        "triton": ("{sglang}/triton", "~/.triton"),
        "flashinfer": ("{sglang}/.cache/flashinfer", "{sglang}/flashinfer/autotune", "~/.cache/flashinfer"),
        "deep_gemm": ("{sglang}/deep_gemm", "~/.cache/deep_gemm"),
    },
    never=(*_COMMON_NEVER, "gpu_p2p_access_cache_for_*", "locks", "runs"),
    env={},  # SGLang points its compilers into its cache root itself; moving it re-downloads FA3
)

PROFILES = {"vllm": VLLM, "sglang": SGLANG}


@dataclass
class Report:
    """What one compile_cache did: recorded, never guessed. Sizes in MB (10**6 bytes), times in
    seconds. `restore` is "missing" (no archive), "restored", "failed" or "not run".
    `groups[name]` counts files per group: restored, new and changed (new/changed: since the
    restore, over every save)."""

    archive: str
    folders: list[str]
    skipped_groups: list[str]
    env: dict[str, str]  # the variables this cache set
    restore: str = "not run"
    restore_error: str | None = None
    restored_files: int = 0
    restored_mb: float = 0.0
    restore_s: float | None = None
    refused_files: int = 0  # archive members outside the folders, in `never`, or behind a link
    existing_files: int = 0  # archive members whose file already existed: kept, not overwritten
    new_files: int = 0
    changed_files: int = 0
    saved: bool = False
    saved_files: int = 0
    saved_mb: float = 0.0
    save_s: float | None = None
    save_error: str | None = None
    replaced_other_writer: bool = False  # the archive changed since this run read it
    groups: dict[str, dict[str, int]] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self))


def compile_cache(
    location: str,
    engine: str | Profile,
    key: str,
    *,
    save: bool = True,
    skip: Iterable[str] = (),
) -> CompileCache:
    """A `CompileCache` for `<location>/<key>.tar`; use it in a with block, or call `restore()`,
    `save()` and `close()` yourself.

    location: `hf://buckets/<you>/<bucket>[/prefix]`, or any fsspec path (a local folder, a
        mounted bucket).
    engine: "vllm", "sglang", or a `Profile`.
    key: the archive name inside `location`, e.g. from `default_key`.
    save: False restores only; `save()` then does nothing.
    skip: profile groups to leave out, e.g. {"deep_gemm"}.

    Restore before the server starts: the environment and the cache files must be there when
    it boots. Needs `saturate[hf]` and a token that can write to the bucket for `hf://`.
    """
    profile = _profile(engine)
    if not key or key.startswith("/") or ".." in key.split("/"):
        raise ValueError(f"compile_cache() requires a relative key without '..', not {key!r}")
    skip = sorted(set(skip))
    unknown = [name for name in skip if name not in profile.groups]
    if unknown:
        raise ValueError(f"compile_cache() requires skip= names from {sorted(profile.groups)}, not {unknown}")
    return CompileCache(f"{location.rstrip('/')}/{key}.tar", profile, save=save, skip=skip)


class CompileCache:
    """One archive's restore and saves. A with block calls `restore()` on enter and, on a clean
    exit, `save()`; `close()` always runs."""

    def __init__(self, archive: str, profile: Profile, *, save: bool = True, skip: Iterable[str] = ()):
        self.archive, self.profile, self.save_enabled = archive, profile, save
        self.skip = sorted(skip)
        self.report = Report(archive=archive, folders=[], skipped_groups=self.skip, env={})
        self._tmp: str | None = None
        self._folders: dict[str, str] = {}  # folder -> group
        self._baseline: dict[str, int] = {}  # path -> size at the last restore or save
        self._remote: tuple = ()  # the archive's stamp when last read or written; () = none there
        self._used = False

    def __enter__(self) -> CompileCache:
        self.restore()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            if exc_type is None:  # an exception in the block skips the save
                self.save()
        finally:
            self.close()

    def restore(self) -> None:
        """Set the environment, then download and unpack the archive if it exists. A cache is
        used once: restore, save as often as you like, close."""
        if self._used:
            raise RuntimeError("compile_cache: restore() requires a new cache; this one was already used")
        self._used = True
        self._tmp = tempfile.mkdtemp(prefix="compile-cache-")
        try:
            unset = {name: value for name, value in self.profile.env.items() if not os.environ.get(name)}
            set_here = {name: _expand(value) for name, value in unset.items()}
            os.environ.update(set_here)
            self.report.env = set_here
            for name in self.profile.env:
                log.info(f"compile_cache: {name}={os.environ[name]}")
            self._folders = _folders(self.profile, self.skip)
            self.report.folders = sorted(self._folders)
            self._restore()
            self._baseline = _manifest(self._folders, self.profile.never)
        except BaseException:  # a failed start must not leave the environment or files behind
            self.close()
            raise

    def save(self) -> None:
        """Save if a cache file is new or changed since the last restore or save (e.g. right
        after the engine is ready; the save at the end then compares with this one)."""
        if self._tmp is None:
            raise RuntimeError("compile_cache: save() requires an open cache (inside its with block)")
        if not self.save_enabled:
            return
        if self.report.restore == "failed":
            log.warning(
                f"compile_cache: not saving {self.archive}: it could not be read at the start, and "
                "this run's files could replace it (if it is broken, delete it to start cold)"
            )
            return
        self._save()

    def close(self) -> None:
        """Remove the temporary folder and the environment variables this cache set."""
        if self._tmp is None:
            return
        shutil.rmtree(self._tmp, ignore_errors=True)
        self._tmp = None
        for name in self.report.env:
            os.environ.pop(name, None)

    def _restore(self) -> None:
        assert self._tmp is not None
        started = time.monotonic()
        local = os.path.join(self._tmp, "restore.tar")
        counts: dict = {"written": 0, "refused": 0, "existing": 0, "by_group": {}}
        report = self.report
        try:
            fs, path = fsspec.url_to_fs(self.archive)
            self._remote = _stamp(fs, path)  # any error but "not found" counts as failed
            if not self._remote:
                report.restore = "missing"
                log.info(f"compile_cache: no archive at {self.archive}; the engine compiles from scratch")
                return
            fs.get_file(path, local)
            download_s = time.monotonic() - started
            size = os.path.getsize(local)
            _unpack(local, self._folders, self.profile.never, counts)
            report.restore = "restored"
        except Exception as e:
            report.restore = "failed"
            report.restore_error = f"{type(e).__name__}: {e}"
            log.warning(
                f"compile_cache: could not restore {self.archive} ({report.restore_error}); "
                "the engine compiles what is missing, and this run will not save"
            )
        finally:
            if os.path.exists(local):
                os.remove(local)
        # recorded even after a failure part-way: those files are on disk
        report.restored_files, report.refused_files = counts["written"], counts["refused"]
        report.existing_files = counts["existing"]
        for group, n in counts["by_group"].items():
            report.groups.setdefault(group, _group_counts())["restored"] = n
        if report.restore != "restored":
            return
        report.restored_mb = round(size / 1e6, 1)
        report.restore_s = round(time.monotonic() - started, 1)
        log.info(
            f"compile_cache: restored {report.restored_mb:,.0f} MB, {report.restored_files:,} files in "
            f"{report.restore_s:.1f} s (download {download_s:.1f} s) from {self.archive}"
        )
        if report.refused_files:
            log.warning(
                f"compile_cache: refused {report.refused_files:,} archive members: outside this run's "
                "folders (or in a skipped group), in `never`, or links"
            )
        if report.existing_files:
            log.info(f"compile_cache: kept {report.existing_files:,} files that already existed")

    def _save(self) -> None:
        assert self._tmp is not None
        after = _manifest(self._folders, self.profile.never)
        new = [path for path in after if path not in self._baseline]
        changed = [
            path for path, size in after.items() if path in self._baseline and self._baseline[path] != size
        ]
        if not new and not changed:
            log.info("compile_cache: no new or changed files in the cache folders; archive not saved")
            return
        started = time.monotonic()
        local = os.path.join(self._tmp, "save.tar")
        try:
            _pack(after, local)
            fs, path = fsspec.url_to_fs(self.archive)
            remote = _stamp(fs, path)
            if remote and remote != self._remote:
                self.report.replaced_other_writer = True
                log.warning(
                    f"compile_cache: another run saved {self.archive} since this one read it; "
                    "replacing it, so that run's new files are lost (one writer per key)"
                )
            _ensure_parent(fs, path)
            _put(fs, local, path)
            self._remote = _stamp(fs, path)
        except Exception as e:
            self.report.save_error = f"{type(e).__name__}: {e}"
            log.warning(f"compile_cache: could not save {self.archive} ({self.report.save_error})")
            return
        finally:
            size = os.path.getsize(local) if os.path.exists(local) else 0
            if os.path.exists(local):
                os.remove(local)
        report = self.report
        report.save_error = None
        report.new_files += len(new)
        report.changed_files += len(changed)
        for kind, paths in (("new", new), ("changed", changed)):
            for path in paths:
                report.groups.setdefault(_group_of(path, self._folders), _group_counts())[kind] += 1
        report.saved = True
        report.saved_files = len(after)
        report.saved_mb = round(size / 1e6, 1)
        report.save_s = round(time.monotonic() - started, 1)
        log.info(
            f"compile_cache: saved {report.saved_mb:,.0f} MB, {len(after):,} files "
            f"({len(new):,} new, {len(changed):,} changed) in {report.save_s:.1f} s to {self.archive}"
        )
        self._baseline = after


def _group_counts() -> dict[str, int]:
    return {"restored": 0, "new": 0, "changed": 0}


# --- the bucket ---


def _is_hf(fs) -> bool:
    protocols = fs.protocol if isinstance(fs.protocol, tuple) else (fs.protocol,)
    return "hf" in protocols


def _stamp(fs, path: str) -> tuple:
    """What tells one version of the archive from another (size, time, hash), read fresh: () when
    it is not there. Any other error is raised, so a failed lookup never reads as "missing"."""
    fs.invalidate_cache()  # HfFileSystem answers info() from a listing it cached earlier
    try:
        info = fs.info(path)
    except FileNotFoundError:
        return ()
    return tuple(str(info.get(k)) for k in ("size", "mtime", "created", "xet_hash", "uploaded_at"))


def _put(fs, local: str, path: str) -> None:
    """An hf:// upload appears whole; elsewhere (a local folder, a mounted bucket) write a
    unique temporary name and rename it, so another run never reads half an archive."""
    if _is_hf(fs):
        fs.put_file(local, path)
        return
    partial = f"{path}.cc-{uuid.uuid4().hex}"
    try:
        fs.put_file(local, partial)
        fs.mv(partial, path)
    finally:
        if fs.exists(partial):
            fs.rm(partial)


def _ensure_parent(fs, path: str) -> None:
    """A missing bucket is created private (as saturate.sink does for outputs); a local folder
    is created."""
    if _is_hf(fs):
        parts = path.split("/")
        if parts[0] == "buckets" and len(parts) >= 3:
            from huggingface_hub import HfApi

            HfApi().create_bucket(f"{parts[1]}/{parts[2]}", private=True, exist_ok=True)
            fs.invalidate_cache()  # a read that found no bucket is cached as missing
        return
    fs.makedirs(os.path.dirname(path), exist_ok=True)


# --- the cache folders on disk ---


def _user() -> str:
    try:
        return getpass.getuser()
    except Exception:  # no user name in the environment or the password database
        return "root"


def _path(path: str) -> str:
    return os.path.normpath(os.path.abspath(os.path.expanduser(path)))


def _expand(template: str) -> str:
    """A profile folder: its `{...}` names filled in, then made absolute."""
    roots = {
        "user": _user(),
        "tmp": tempfile.gettempdir(),
        "vllm": os.environ.get("VLLM_CACHE_ROOT") or "~/.cache/vllm",
        "sglang": os.environ.get("SGLANG_CACHE_DIR") or "~/.cache/sglang",
    }
    return _path(template.format(**roots))


def _within(path: str, folder: str) -> bool:
    return path == folder or path.startswith(folder.rstrip("/") + "/")


def _folders(profile: Profile, skip: Iterable[str] = ()) -> dict[str, str]:
    """folder -> group for the profile's groups not skipped; a folder inside another is listed
    once, under the outer folder's group."""
    found: dict[str, str] = {}
    for group, templates in profile.groups.items():
        if group in skip:
            continue
        for template in templates:
            if template.startswith("$"):
                value = os.environ.get(template[1:])
                if value:
                    found.setdefault(_path(value), group)
            else:
                found.setdefault(_expand(template), group)
    kept: dict[str, str] = {}
    for path in sorted(found):
        if not any(_within(path, folder) for folder in kept):
            kept[path] = found[path]
    return kept


def _group_of(path: str, folders: dict[str, str]) -> str:
    return next((group for folder, group in folders.items() if _within(path, folder)), "?")


def _guarded() -> list[str]:
    """Folders never archived or restored, whatever the profile says: the Hugging Face cache
    (model weights and the token), by default and where HF_HOME / HF_HUB_CACHE move it."""
    homes = ["~/.cache/huggingface", os.environ.get("HF_HOME"), os.environ.get("HF_HUB_CACHE")]
    return [_path(home) for home in homes if home]


def _skipped(rel: str, never: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatchcase(part, pattern) for pattern in never for part in rel.split("/"))


def _manifest(folders: dict[str, str], never: tuple[str, ...]) -> dict[str, int]:
    """path -> size of every regular file in the folders (symbolic links are left out)."""
    guarded = _guarded()
    files: dict[str, int] = {}
    for folder in folders:
        for dirpath, dirnames, filenames in os.walk(folder):
            if any(_within(dirpath, g) for g in guarded):
                dirnames[:] = []
                continue
            for name in filenames:
                path = os.path.join(dirpath, name)
                if _skipped(os.path.relpath(path, folder), never) or os.path.islink(path):
                    continue
                try:
                    files[path] = os.path.getsize(path)
                except OSError:  # removed since the listing
                    continue
    return files


def _arcname(path: str) -> str:
    """Paths under the home folder are stored relative to it, so a different home still works."""
    home = os.path.expanduser("~")
    if _within(path, home):
        return "home/" + os.path.relpath(path, home)
    return "root/" + path.lstrip("/")


def _target(arcname: str) -> str | None:
    prefix, _, rest = arcname.partition("/")
    if prefix == "home" and rest:
        return os.path.normpath(os.path.join(os.path.expanduser("~"), rest))
    if prefix == "root" and rest:
        return os.path.normpath("/" + rest)
    return None


def _pack(files: dict[str, int], tar_path: str) -> None:
    """One tar of the files, in path order, each stored whole (a hard link would come back as a
    link, which restore refuses). A file removed since the listing is left out; any other error
    (a file that shrinks while it is read) fails the whole save, because tarfile would leave the
    archive cut short at that file."""
    with tarfile.open(tar_path, "w", dereference=True) as tar:
        for path in sorted(files):
            try:
                tar.add(path, arcname=_arcname(path), recursive=False)
            except FileNotFoundError:  # raised before anything of this file is written
                continue


def _unpack(tar_path: str, folders: dict[str, str], never: tuple[str, ...], counts: dict) -> None:
    """Write the archive's regular files that belong in the folders. Refused: links, members
    outside the folders, `never` files and the Hugging Face cache. A file that already exists
    (e.g. one the image ships) is kept. Each file is written under a temporary name and renamed,
    so a broken archive leaves no partial file. Counts go into `counts` (written, refused,
    existing, by_group) as it goes, so they stay right when a broken archive stops it part-way."""
    guarded = _guarded()
    with tarfile.open(tar_path) as tar:
        for member in tar:
            if member.isdir():
                continue
            target = _target(member.name) if member.isfile() else None
            folder = next((f for f in folders if target and _within(target, f)), None)
            if (
                target is None
                or folder is None
                or _skipped(os.path.relpath(target, folder), never)
                or any(_within(target, g) for g in guarded)
            ):
                counts["refused"] += 1
                continue
            if os.path.lexists(target):
                counts["existing"] += 1
                continue
            source = tar.extractfile(member)
            if source is None:
                counts["refused"] += 1
                continue
            parent = os.path.dirname(target)
            os.makedirs(parent, exist_ok=True)
            handle, partial = tempfile.mkstemp(prefix=".cc-", dir=parent)
            try:
                with source, os.fdopen(handle, "wb") as out:
                    shutil.copyfileobj(source, out)
                os.chmod(partial, member.mode & 0o755)
                os.utime(partial, (member.mtime, member.mtime))  # build tools (ninja) compare mtimes
                os.replace(partial, target)
            finally:
                if os.path.exists(partial):
                    os.remove(partial)
            counts["written"] += 1
            group = folders[folder]
            counts["by_group"][group] = counts["by_group"].get(group, 0) + 1


# --- the key ---


def default_key(
    engine: str,
    model: str,
    *,
    revision: str | None = None,
    version: str | None = None,
    image: str | None = None,
    gpu: str | None = None,
    extra: Iterable[str] = (),
) -> str:
    """A coarse archive key: `<engine>/<model>[@<revision>]/<gpu>/<engine>-<version>[/<image>]
    [/<extra>...]`, e.g. `vllm/Qwen--Qwen3-8B/NVIDIA-L4-sm89-x1/vllm-0.32.0`.

    version: the engine's installed version. By default it is read from this Python; for
        SGLang also from the first `python3` on PATH that has it (as `Engine` finds SGLang).
        Pass it when the engine lives elsewhere (vLLM in a venv of its own), and call this
        after installing a different build.
    image: the container image tag or digest. Recommended: it covers the torch and FlashInfer
        versions inside the image, and keeps archives for different images apart.
    gpu: the GPU part; by default `<name>-sm<capability>-x<count>` from nvidia-smi.
    extra: anything else to keep apart, e.g. "tp2" or "nvfp4".
    """
    version = version or _installed_version(engine)
    name = model + (f"@{revision}" if revision else "")
    parts = [engine, _slug(name), _slug(gpu) if gpu else _gpu(), f"{engine}-{_slug(version)}"]
    if image:
        parts.append(_slug(image))
    parts += [_slug(e) for e in extra]
    return "/".join(parts)


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._@-]+", "-", text.replace("/", "--")).strip("-")


def _gpu() -> str:
    """`<name>-sm<capability>-x<count>` of the GPUs nvidia-smi lists (it ignores
    CUDA_VISIBLE_DEVICES; pass gpu= to default_key when that matters)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError) as e:
        raise ValueError(
            f"default_key() requires gpu= when nvidia-smi gives no GPU ({type(e).__name__})"
        ) from e
    rows = [line.split(",") for line in out.splitlines() if line.strip()]
    if not rows or len(rows[0]) < 2:
        raise ValueError(f"default_key() requires gpu= when nvidia-smi gives no GPU (output {out[:80]!r})")
    name, capability = rows[0][0].strip(), rows[0][1].strip().replace(".", "")
    return f"{_slug(name)}-sm{capability}-x{len(rows)}"


_VERSION = "import importlib.metadata as m, sys; print(m.version(sys.argv[1]))"


def _installed_version(engine: str) -> str:
    try:
        return importlib.metadata.version(engine)
    except importlib.metadata.PackageNotFoundError:
        pass
    if engine == "sglang":  # the image's own Python, which Engine also starts SGLang with
        from saturate.engine import python_with

        try:
            asked = subprocess.run(
                [python_with("sglang"), "-c", _VERSION, "sglang"], capture_output=True, text=True, timeout=30
            )
            if asked.returncode == 0 and asked.stdout.strip():
                return asked.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    raise ValueError(f"default_key() requires version= when {engine} is not installed in this Python")


def _profile(engine: str | Profile) -> Profile:
    if isinstance(engine, Profile):
        return engine
    if engine not in PROFILES:
        raise ValueError(
            f"compile_cache() requires engine= from {sorted(PROFILES)} or a Profile, not {engine!r}"
        )
    return PROFILES[engine]
