"""Lab helpers: not part of the contract; may change or be removed in any release, without
deprecation.

`compile_cache` keeps a serving engine's start-up caches (compiled graphs, Triton and FlashInfer
kernels, autotune results) as one archive in a Bucket you own. Put it around the code that
starts the engine:

    from saturate import Engine, pump
    from saturate.labs.compile_cache import compile_cache, default_key

    key = default_key("vllm", MODEL)
    with compile_cache("hf://buckets/<you>/jobs-cache", engine="vllm", key=key):
        with Engine(MODEL, engine="vllm") as endpoint:
            pump(...)

On enter, it sets the engine profile's environment variables (only those you have not set),
downloads `<location>/<key>.tar` if it exists, and unpacks it into the engine's cache folders.
On a clean exit, it lists the cache folders again (path and size of each file) and uploads a
new archive of all of them only if a file is new or changed size. An exception in the block
skips the save. A failed download or upload logs a warning and the run goes on: the cache only
saves time.

The archive holds code that the engine loads and runs (.py, .so, cubins). Keep it in your own
private bucket (a missing bucket is created private), and never restore an archive that other
people can write to.

The key is coarse on purpose (engine, version, model, GPU). Each engine keys its own cache
folders by hashes that this helper does not compute. A key that is too coarse costs a
recompile and a larger archive, not wrong output.
"""

from __future__ import annotations

import dataclasses
import fnmatch
import getpass
import importlib.metadata
import logging
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import warnings
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field

import fsspec

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Profile:
    """What compile_cache does for one engine.

    folders: cache folders to archive when they exist. A path starts with `~` or `/`; `{user}`
        is the user name, `{vllm}` and `{sglang}` are the engines' cache roots
        (VLLM_CACHE_ROOT, SGLANG_CACHE_DIR, or their defaults).
    never: files never archived or restored. A pattern without `/` is matched against every
        part of a file's path inside its folder (`*.lock`, `runs`); a pattern with `/` against
        the start of that path (`assets/*`).
    env: environment variables set before the engine starts, unless already set. Their values
        are folders, and are archived too.
    log_lines: name -> regular expression; `scan_log` collects the lines that match.
    """

    folders: tuple[str, ...]
    never: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    log_lines: dict[str, str] = field(default_factory=dict)


# Never in any archive: locks, temporary files, half-written staging folders.
_COMMON_NEVER = ("*.lock", "*.tmp", "tmp.*", ".staging-*")

VLLM = Profile(
    folders=(
        # torch_compile_cache, torch_aot_compile, flashinfer_autotune_cache, deep_gemm, startup_plan
        "{vllm}",
        "~/.triton",
        "~/.cache/flashinfer",
        "/tmp/torchinductor_{user}",
        "~/.nv/ComputeCache",
    ),
    never=(*_COMMON_NEVER, "assets/*"),
    # torch's first inductor autotuner otherwise points TRITON_CACHE_DIR at the inductor cache
    # of that moment, which moves between a cold and a warm run: the warm run then misses.
    env={"TRITON_CACHE_DIR": "~/.triton/cache"},
    log_lines={
        "aot_loaded": r"Directly load AOT compilation",
        "aot_saved": r"[Ss]aved AOT",
        "torch_compile_s": r"torch\.compile takes ([\d.]+) s in total",
        "profiling_run_s": r"Initial profiling/warmup run took ([\d.]+) s",
        "jit_warmup_s": r"JIT kernel warmup finished in ([\d.]+)",
        "autotune_loaded": r"Loaded (\d+) configs",
        "autotune_tuning": r"Tuning (\S+)",
        "startup_plan": r"Applying persisted startup plan",
        "graph_capture_s": r"Graph capturing finished in ([\d.]+) sec",
        "engine_init_s": r"init engine .*? took ([\d.]+) s",
        "cubin_reload_failed": r"Failed to reload cubin file",
        "jit_during_inference": r"JIT compilation during inference",
    },
)

SGLANG = Profile(
    # Subfolders of the SGLang cache root, named one by one: the root also holds files that
    # must never move to another machine (kernels.lock from the image, gpu_p2p_* per host,
    # model_streamer, ple, paged_experts). The last five folders are where older images write.
    folders=(
        "{sglang}/jit",
        "{sglang}/deep_gemm",
        "{sglang}/cute_aot",
        "{sglang}/triton",
        "{sglang}/inductor",
        "{sglang}/nv",
        "{sglang}/.cache/flashinfer",
        "{sglang}/flashinfer/autotune",
        "{sglang}/torch_compile_cache",
        "{sglang}/tilelang",
        "{sglang}/humming/cache",
        "~/.triton",
        "~/.cache/flashinfer",
        "~/.cache/deep_gemm",
        "/tmp/torchinductor_{user}",
        "~/.nv/ComputeCache",
    ),
    never=(*_COMMON_NEVER, "gpu_p2p_access_cache_for_*", "locks", "runs"),
    env={},  # SGLang points its compilers into its cache root itself; moving it re-downloads FA3
    log_lines={
        "ready": r"The server is fired up and ready to roll",
        "capture_s": r"Capture .*?elapsed[=:]\s*([\d.]+)",
        "deep_gemm_jit": r"JIT Compiling",
        "serving_compile": r"Triton kernel '(\S+)' took [\d.]+ s to compile after serving started",
    },
)

PROFILES = {"vllm": VLLM, "sglang": SGLANG}


@dataclass
class Report:
    """What one compile_cache block did. Sizes in MB (10**6 bytes), times in seconds."""

    archive: str
    folders: list[str]
    env: dict[str, str]  # the variables this block set
    restored_files: int = 0
    restored_mb: float = 0.0
    restore_s: float | None = None  # None: nothing restored
    refused_files: int = 0  # archive members outside the profile's folders, or in `never`
    new_files: int = 0
    changed_files: int = 0
    saved: bool = False
    saved_files: int = 0
    saved_mb: float = 0.0
    save_s: float | None = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


_warned = False


def _warn_once() -> None:
    global _warned
    if not _warned:
        _warned = True
        warnings.warn(
            "saturate.labs.compile_cache is a lab helper: not part of the contract; it may "
            "change or be removed in any release, without deprecation",
            FutureWarning,
            stacklevel=3,
        )


def compile_cache(location: str, engine: str | Profile, key: str, *, save: bool = True):
    """Restore `<location>/<key>.tar` into the engine's cache folders on enter; on a clean exit,
    save a new archive if a cache file is new or changed size. Yields a `Report`.

    location: `hf://buckets/<you>/<bucket>[/prefix]`, or any fsspec path (a local folder, a
        mounted bucket).
    engine: "vllm", "sglang", or a `Profile`.
    key: the archive name inside `location`, e.g. from `default_key`.
    save: False restores only.
    """
    _warn_once()
    profile = _profile(engine)
    if not key or key.startswith("/") or ".." in key.split("/"):
        raise ValueError(f"compile_cache: key must be a relative name without '..', got {key!r}")
    return _session(f"{location.rstrip('/')}/{key}.tar", profile, save)


@contextmanager
def _session(archive: str, profile: Profile, save: bool) -> Generator[Report]:
    set_here = {name: _expand(value) for name, value in profile.env.items() if not os.environ.get(name)}
    os.environ.update(set_here)
    try:
        folders = _folders(profile)
        report = Report(archive=archive, folders=folders, env=set_here)
        for name in profile.env:
            log.info(f"compile_cache: {name}={os.environ[name]}")
        with tempfile.TemporaryDirectory(prefix="compile-cache-") as tmp:
            _restore(archive, folders, profile.never, tmp, report)
            before = _manifest(folders, profile.never)
            yield report
            if save:
                _save(archive, folders, profile.never, before, tmp, report)
    finally:
        for name in set_here:
            os.environ.pop(name, None)


def _restore(archive: str, folders: list[str], never: tuple[str, ...], tmp: str, report: Report) -> None:
    started = time.monotonic()
    local = os.path.join(tmp, "restore.tar")
    try:
        fs, path = fsspec.url_to_fs(archive)
        if not fs.exists(path):
            log.info(f"compile_cache: no archive at {archive}; the engine compiles from scratch")
            return
        fs.get_file(path, local)
        download_s = time.monotonic() - started
        size = os.path.getsize(local)
        files, refused = _unpack(local, folders, never)
    except Exception as e:
        log.warning(
            f"compile_cache: could not restore {archive} ({type(e).__name__}: {e}); "
            "the engine compiles what is missing"
        )
        return
    finally:
        if os.path.exists(local):
            os.remove(local)
    report.restored_files, report.refused_files = files, refused
    report.restored_mb = round(size / 1e6, 1)
    report.restore_s = round(time.monotonic() - started, 1)
    log.info(
        f"compile_cache: restored {report.restored_mb:,.0f} MB, {files:,} files in "
        f"{report.restore_s:.1f} s (download {download_s:.1f} s) from {archive}"
    )
    if refused:
        log.warning(f"compile_cache: skipped {refused:,} archive members outside the cache folders")


def _save(
    archive: str,
    folders: list[str],
    never: tuple[str, ...],
    before: dict[str, int],
    tmp: str,
    report: Report,
) -> None:
    after = _manifest(folders, never)
    report.new_files = sum(1 for path in after if path not in before)
    report.changed_files = sum(1 for path, size in after.items() if path in before and before[path] != size)
    if not report.new_files and not report.changed_files:
        log.info("compile_cache: no new or changed files in the cache folders; archive not saved")
        return
    started = time.monotonic()
    local = os.path.join(tmp, "save.tar")
    try:
        _pack(after, local)
        fs, path = fsspec.url_to_fs(archive)
        _ensure_parent(fs, path)
        fs.put_file(local, path)
    except Exception as e:
        log.warning(f"compile_cache: could not save {archive} ({type(e).__name__}: {e})")
        return
    report.saved = True
    report.saved_files = len(after)
    report.saved_mb = round(os.path.getsize(local) / 1e6, 1)
    report.save_s = round(time.monotonic() - started, 1)
    log.info(
        f"compile_cache: saved {report.saved_mb:,.0f} MB, {len(after):,} files "
        f"({report.new_files:,} new, {report.changed_files:,} changed) in {report.save_s:.1f} s "
        f"to {archive}"
    )


def _ensure_parent(fs, path: str) -> None:
    """A missing bucket is created private; a local folder is created."""
    protocols = fs.protocol if isinstance(fs.protocol, tuple) else (fs.protocol,)
    if "hf" in protocols:
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


def _expand(path: str) -> str:
    roots = {
        "user": _user(),
        "vllm": os.environ.get("VLLM_CACHE_ROOT") or "~/.cache/vllm",
        "sglang": os.environ.get("SGLANG_CACHE_DIR") or "~/.cache/sglang",
    }
    return os.path.normpath(os.path.expanduser(path.format(**roots)))


def _within(path: str, folder: str) -> bool:
    return path == folder or path.startswith(folder.rstrip("/") + "/")


def _folders(profile: Profile) -> list[str]:
    """The profile's folders as absolute paths, a folder inside another listed once."""
    paths = [_expand(p) for p in profile.folders]
    paths += [_expand(os.environ[name]) for name in profile.env if os.environ.get(name)]
    kept: list[str] = []
    for path in sorted(set(paths)):
        if not any(_within(path, folder) for folder in kept):
            kept.append(path)
    return kept


def _guarded() -> list[str]:
    """Folders never archived or restored, whatever the profile says: the Hugging Face cache
    holds model weights and the token."""
    homes = [os.environ.get("HF_HOME") or "~/.cache/huggingface", os.environ.get("HF_HUB_CACHE")]
    return [os.path.normpath(os.path.expanduser(home)) for home in homes if home]


def _skipped(rel: str, never: tuple[str, ...]) -> bool:
    parts = rel.split("/")
    for pattern in never:
        if "/" in pattern:
            if fnmatch.fnmatchcase(rel, pattern) or fnmatch.fnmatchcase(rel, pattern + "/*"):
                return True
        elif any(fnmatch.fnmatchcase(part, pattern) for part in parts):
            return True
    return False


def _manifest(folders: list[str], never: tuple[str, ...]) -> dict[str, int]:
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
    with tarfile.open(tar_path, "w") as tar:
        for path in sorted(files):
            try:
                tar.add(path, arcname=_arcname(path), recursive=False)
            except OSError:  # removed since the listing
                continue


def _unpack(tar_path: str, folders: list[str], never: tuple[str, ...]) -> tuple[int, int]:
    """Write the archive's regular files that belong in the folders; return (written, refused).
    Links, members outside the folders, `never` files and the Hugging Face cache are refused."""
    guarded = _guarded()
    written = refused = 0
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
                or os.path.islink(target)
            ):
                refused += 1
                continue
            os.makedirs(os.path.dirname(target), exist_ok=True)
            source = tar.extractfile(member)
            if source is None:
                refused += 1
                continue
            with source, open(target, "wb") as out:
                shutil.copyfileobj(source, out)
            os.chmod(target, member.mode & 0o755)
            os.utime(target, (member.mtime, member.mtime))  # build tools (ninja) compare mtimes
            written += 1
    return written, refused


# --- the key ---


def default_key(
    engine: str,
    model: str,
    *,
    revision: str | None = None,
    version: str | None = None,
    image: str | None = None,
    extra: Iterable[str] = (),
) -> str:
    """A coarse archive key: `<engine>/<model>[@<revision>]/<gpu>/<engine>-<version>[/<image>]
    [/<extra>...]`, e.g. `vllm/Qwen--Qwen3-8B/NVIDIA-L4-sm89-x1/vllm-0.32.0`.

    version: the engine's installed version. By default it is read from this Python, else
        from the first `python3` on PATH that has the engine (as `Engine` finds SGLang). Pass
        it when the engine lives in an environment of its own, or after installing a
        different build (call this after the install).
    image: the container image tag or digest; recommended when the engine came with the image.
    extra: anything else that changes compiled code, e.g. "tp2" for tensor parallelism.
    """
    version = version or _installed_version(engine)
    name = model + (f"@{revision}" if revision else "")
    parts = [engine, _slug(name), _gpu(), f"{engine}-{_slug(version)}"]
    if image:
        parts.append(_slug(image))
    parts += [_slug(e) for e in extra]
    return "/".join(parts)


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._@-]+", "-", text.replace("/", "--")).strip("-")


def _gpu() -> str:
    """`<name>-sm<capability>-x<count>` of the visible GPUs, or `no-gpu`."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return "no-gpu"
    rows = [line.split(",") for line in out.splitlines() if line.strip()]
    if not rows or len(rows[0]) < 2:
        return "no-gpu"
    name, capability = rows[0][0].strip(), rows[0][1].strip().replace(".", "")
    return f"{_slug(name)}-sm{capability}-x{len(rows)}"


_VERSION = "import importlib.metadata as m, sys; print(m.version(sys.argv[1]))"


def _installed_version(engine: str) -> str:
    if engine not in PROFILES:
        raise ValueError(f"default_key: pass version= for engine {engine!r}")
    try:
        return importlib.metadata.version(engine)
    except importlib.metadata.PackageNotFoundError:
        pass
    from saturate.engine import python_with

    python = python_with(engine)
    if python != sys.executable:
        try:
            asked = subprocess.run(
                [python, "-c", _VERSION, engine], capture_output=True, text=True, timeout=30
            )
            if asked.returncode == 0 and asked.stdout.strip():
                return asked.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    raise ValueError(f"default_key: no installed {engine} found; pass version=")


# --- logs ---


def scan_log(engine: str | Profile, lines: Iterable[str] | str) -> dict[str, list[str]]:
    """The profile's log lines found in `lines`: name -> one entry per matching line (the
    first group of the expression, or the whole match). Counts are the list lengths."""
    profile = _profile(engine)
    if isinstance(lines, str):
        lines = lines.splitlines()
    patterns = {name: re.compile(p) for name, p in profile.log_lines.items()}
    found: dict[str, list[str]] = {name: [] for name in patterns}
    for line in lines:
        for name, pattern in patterns.items():
            match = pattern.search(line)
            if match:
                found[name].append((match.group(1) if match.groups() else None) or match.group(0))
    return found


def _profile(engine: str | Profile) -> Profile:
    if isinstance(engine, Profile):
        return engine
    if engine not in PROFILES:
        raise ValueError(f"compile_cache: no profile for engine {engine!r}; use one of {sorted(PROFILES)}")
    return PROFILES[engine]
