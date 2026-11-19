"""saturate.labs.compile_cache on fake cache folders and a local folder as the bucket."""

import os
import subprocess
import sys
import tarfile

import pytest

from saturate.labs import compile_cache as cc


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A fake home folder; the engine roots and the Hugging Face cache follow it."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    for name in ("VLLM_CACHE_ROOT", "SGLANG_CACHE_DIR", "HF_HOME", "HF_HUB_CACHE", "TRITON_CACHE_DIR"):
        monkeypatch.delenv(name, raising=False)
    return home


@pytest.fixture
def profile(home):
    return cc.Profile(
        folders=("~/.cache/engine", "~/.triton"),
        never=("*.lock", "locks", "runs", "assets/*", "gpu_p2p_*"),
        env={"TRITON_CACHE_DIR": "~/.triton/cache"},
    )


def write(path, text="x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def members(archive):
    with tarfile.open(archive) as tar:
        return sorted(m.name for m in tar.getmembers() if m.isfile())


def test_cold_save_then_warm_restore_saves_nothing(tmp_path, home, profile):
    bucket = tmp_path / "bucket"
    with cc.compile_cache(str(bucket), profile, "vllm/m/gpu/v1") as report:
        assert report.restore_s is None
        write(home / ".cache/engine/hash1/graph.py", "graph")
        kernel = write(home / ".triton/cache/abc/kernel.cubin", "cubin")
        os.utime(kernel, (1_000_000, 1_000_000))
    archive = bucket / "vllm/m/gpu/v1.tar"
    assert report.saved and report.new_files == 2 and archive.exists()
    assert members(archive) == ["home/.cache/engine/hash1/graph.py", "home/.triton/cache/abc/kernel.cubin"]

    for folder in (home / ".cache", home / ".triton"):  # a fresh machine
        for path in sorted(folder.rglob("*"), reverse=True):
            path.unlink() if path.is_file() else path.rmdir()
    with cc.compile_cache(str(bucket), profile, "vllm/m/gpu/v1") as report:
        assert (home / ".cache/engine/hash1/graph.py").read_text() == "graph"
        assert kernel.read_text() == "cubin"
        assert kernel.stat().st_mtime == 1_000_000  # kept: build tools compare mtimes
    assert report.restored_files == 2 and report.restore_s is not None
    assert not report.saved and report.new_files == 0


def test_new_or_changed_file_saves_the_union(tmp_path, home, profile):
    bucket = tmp_path / "bucket"
    with cc.compile_cache(str(bucket), profile, "k"):
        write(home / ".cache/engine/a.py", "a")
        write(home / ".cache/engine/b.json", "{}")
    with cc.compile_cache(str(bucket), profile, "k") as report:
        write(home / ".cache/engine/c.py", "c")  # new
        write(home / ".cache/engine/b.json", '{"tuned": 1}')  # changed size
    assert report.saved and report.new_files == 1 and report.changed_files == 1
    assert members(bucket / "k.tar") == [
        "home/.cache/engine/a.py",
        "home/.cache/engine/b.json",
        "home/.cache/engine/c.py",
    ]


def test_rewritten_file_of_the_same_size_saves_nothing(tmp_path, home, profile):
    bucket = tmp_path / "bucket"
    with cc.compile_cache(str(bucket), profile, "k"):
        write(home / ".cache/engine/autotune.json", "1111")
    with cc.compile_cache(str(bucket), profile, "k") as report:
        write(home / ".cache/engine/autotune.json", "2222")  # rewritten every run
    assert not report.saved


def test_never_files_and_the_hf_cache_stay_out(tmp_path, home, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(home / ".cache/engine/hf"))  # inside a listed folder
    profile = cc.Profile(folders=("~/.cache/engine",), never=cc.SGLANG.never)
    bucket = tmp_path / "bucket"
    with cc.compile_cache(str(bucket), profile, "k"):
        write(home / ".cache/engine/jit/kernel.so")
        write(home / ".cache/engine/kernels.lock")
        write(home / ".cache/engine/gpu_p2p_access_cache_for_0,1.json")
        write(home / ".cache/engine/deep_gemm/locks/a")
        write(home / ".cache/engine/runs/r1/x")
        write(home / ".cache/engine/.staging-1234/half.so")
        write(home / ".cache/engine/x.tmp")
        write(home / ".cache/engine/hf/token", "hf_secret")
        os.symlink(home / ".cache/engine/jit/kernel.so", home / ".cache/engine/link.so")
    assert members(bucket / "k.tar") == ["home/.cache/engine/jit/kernel.so"]


def test_restore_refuses_members_outside_the_folders(tmp_path, home, profile):
    bucket = tmp_path / "bucket"
    bucket.mkdir()
    payload = write(tmp_path / "payload", "evil")
    with tarfile.open(bucket / "k.tar", "w") as tar:
        tar.add(payload, arcname="home/.cache/engine/ok.py")
        tar.add(payload, arcname="home/.bashrc")
        tar.add(payload, arcname="home/.cache/engine/../../.profile")
        tar.add(payload, arcname="home/.cache/engine/assets/big.bin")
        tar.add(payload, arcname="root/etc/evil")
        tar.add(payload, arcname="elsewhere/x")
    with cc.compile_cache(str(bucket), profile, "k", save=False) as report:
        pass
    assert report.restored_files == 1 and report.refused_files == 5
    assert (home / ".cache/engine/ok.py").exists()
    assert not (home / ".bashrc").exists() and not (home / ".profile").exists()
    assert not (home / ".cache/engine/assets").exists()


def test_exception_in_the_block_skips_the_save_and_restores_env(tmp_path, home, profile):
    bucket = tmp_path / "bucket"
    with pytest.raises(RuntimeError), cc.compile_cache(str(bucket), profile, "k"):
        assert os.environ["TRITON_CACHE_DIR"] == str(home / ".triton/cache")
        write(home / ".cache/engine/half.py")
        raise RuntimeError("engine died during boot")
    assert not (bucket / "k.tar").exists()
    assert "TRITON_CACHE_DIR" not in os.environ


def test_env_set_only_when_unset_and_its_folder_archived(tmp_path, home, profile, monkeypatch):
    mine = tmp_path / "my-triton"
    monkeypatch.setenv("TRITON_CACHE_DIR", str(mine))
    bucket = tmp_path / "bucket"
    with cc.compile_cache(str(bucket), profile, "k") as report:
        assert os.environ["TRITON_CACHE_DIR"] == str(mine)
        write(mine / "k.cubin")
    assert report.env == {}
    assert os.environ["TRITON_CACHE_DIR"] == str(mine)
    assert members(bucket / "k.tar") == ["root" + str(mine / "k.cubin")]


def test_unreadable_archive_warns_and_runs_cold(tmp_path, home, profile, caplog):
    bucket = tmp_path / "bucket"
    write(bucket / "k.tar", "not a tar")
    with cc.compile_cache(str(bucket), profile, "k", save=False) as report:
        pass
    assert report.restore_s is None
    assert "could not restore" in caplog.text


def test_engine_profiles_and_bad_keys(home, monkeypatch, tmp_path):
    monkeypatch.setenv("SGLANG_CACHE_DIR", str(tmp_path / "sg"))
    folders = cc._folders(cc.SGLANG)
    assert str(tmp_path / "sg/jit") in folders and str(tmp_path / "sg") not in folders
    with pytest.raises(ValueError):
        cc.compile_cache(str(tmp_path), "tgi", "k")
    with pytest.raises(ValueError):
        cc.compile_cache(str(tmp_path), "vllm", "../k")


def test_vllm_profile_pins_triton_and_names_subfolders(home, monkeypatch, tmp_path):
    (tmp_path / "scratch").mkdir()  # gettempdir() skips a TMPDIR that does not exist
    monkeypatch.setenv("TMPDIR", str(tmp_path / "scratch"))
    monkeypatch.setattr(cc.tempfile, "tempdir", None)  # gettempdir() reads TMPDIR again
    folders = cc._folders(cc.VLLM)
    assert str(home / ".cache/vllm/torch_compile_cache") in folders
    assert str(home / ".cache/vllm") not in folders  # its assets folder stays out
    assert str(home / ".triton/cache") not in folders  # inside ~/.triton, listed once
    assert str(tmp_path / "scratch" / f"torchinductor_{cc._user()}") in folders
    assert cc.VLLM.env == {"TRITON_CACHE_DIR": "~/.triton/cache"}


def test_save_inside_the_block_survives_a_later_exception(tmp_path, home, profile):
    bucket = tmp_path / "bucket"
    with pytest.raises(RuntimeError), cc.compile_cache(str(bucket), profile, "k") as cache:
        write(home / ".cache/engine/boot.py")  # compiled while the engine booted
        cache.save()
        write(home / ".cache/engine/later.py")
        raise RuntimeError("pump failed")
    assert members(bucket / "k.tar") == ["home/.cache/engine/boot.py"]
    assert cache.saved and cache.new_files == 1


def test_exit_save_compares_with_the_save_inside_the_block(tmp_path, home, profile):
    bucket = tmp_path / "bucket"
    with cc.compile_cache(str(bucket), profile, "k") as cache:
        write(home / ".cache/engine/boot.py")
        cache.save()
    assert cache.new_files == 1  # the exit found nothing new: one save
    with cc.compile_cache(str(bucket), profile, "k") as cache:
        cache.save()
        write(home / ".cache/engine/inference.py")  # compiled while serving
    assert cache.new_files == 1
    assert len(members(bucket / "k.tar")) == 2


def test_restore_keeps_files_that_exist(tmp_path, home, profile):
    bucket = tmp_path / "bucket"
    with cc.compile_cache(str(bucket), profile, "k"):
        write(home / ".cache/engine/shipped.so", "from the archive")
    write(home / ".cache/engine/shipped.so", "from the image")
    with cc.compile_cache(str(bucket), profile, "k", save=False) as cache:
        pass
    assert (home / ".cache/engine/shipped.so").read_text() == "from the image"
    assert cache.existing_files == 1 and cache.restored_files == 0


def test_truncated_archive_leaves_no_partial_file(tmp_path, home, profile):
    bucket = tmp_path / "bucket"
    with cc.compile_cache(str(bucket), profile, "k"):
        write(home / ".cache/engine/a.so", "a" * 2000)
        write(home / ".cache/engine/b.so", "b" * 20000)
    archive = bucket / "k.tar"
    archive.write_bytes(archive.read_bytes()[:8000])  # ends inside b.so's data
    for path in (home / ".cache/engine").iterdir():
        path.unlink()
    with cc.compile_cache(str(bucket), profile, "k", save=False):
        pass
    assert sorted(p.name for p in (home / ".cache/engine").iterdir()) == ["a.so"]


def test_archive_moves_to_a_different_home(tmp_path, home, profile, monkeypatch):
    bucket = tmp_path / "bucket"
    with cc.compile_cache(str(bucket), profile, "k"):
        write(home / ".cache/engine/a.py", "a")
    other = tmp_path / "other-home"
    other.mkdir()
    monkeypatch.setenv("HOME", str(other))
    with cc.compile_cache(str(bucket), profile, "k", save=False) as cache:
        pass
    assert (other / ".cache/engine/a.py").read_text() == "a" and cache.restored_files == 1


def test_failed_upload_warns_and_the_block_does_not_raise(tmp_path, home, profile, monkeypatch, caplog):
    def broken(fs, local, path):
        raise OSError("bucket unreachable")

    monkeypatch.setattr(cc, "_put", broken)
    with cc.compile_cache(str(tmp_path / "bucket"), profile, "k") as cache:
        write(home / ".cache/engine/a.py")
    assert not cache.saved and "could not save" in caplog.text


def test_missing_bucket_is_created_private(monkeypatch):
    made = []

    class FakeApi:
        def create_bucket(self, repo_id, private, exist_ok):
            made.append((repo_id, private, exist_ok))

    class FakeHfFs:
        protocol = ("hf",)

        def invalidate_cache(self):
            made.append("invalidated")

    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)
    cc._ensure_parent(FakeHfFs(), "buckets/me/jobs-cache/vllm/m/gpu/v.tar")
    assert made == [("me/jobs-cache", True, True), "invalidated"]


def test_gpu_name_capability_and_count(monkeypatch):
    class Done:
        stdout = "NVIDIA H200, 9.0\nNVIDIA H200, 9.0\n"

    monkeypatch.setattr(cc.subprocess, "run", lambda *a, **k: Done())
    assert cc._gpu() == "NVIDIA-H200-sm90-x2"
    Done.stdout = "garbage\n"
    assert cc._gpu() == "no-gpu"


def test_default_key(monkeypatch):
    monkeypatch.setattr(cc, "_gpu", lambda: "NVIDIA-RTX-PRO-6000-sm120-x1")
    key = cc.default_key(
        "sglang",
        "baidu/Unlimited-OCR-2",
        version="0.5.20.dev10+g8d62f5fb9c",
        image="lmsysorg/sglang:v0.5.20-cu130",
        extra=["tp1"],
    )
    assert key == (
        "sglang/baidu--Unlimited-OCR-2/NVIDIA-RTX-PRO-6000-sm120-x1/"
        "sglang-0.5.20.dev10-g8d62f5fb9c/lmsysorg--sglang-v0.5.20-cu130/tp1"
    )


def test_default_key_without_a_version_raises(monkeypatch):
    monkeypatch.setattr(cc, "_gpu", lambda: "gpu")
    monkeypatch.setattr("saturate.engine.python_with", lambda module: "/nonexistent/python3")
    with pytest.raises(ValueError, match="pass version="):
        cc.default_key("vllm", "m")
    with pytest.raises(ValueError, match="pass version="):
        cc.default_key("llamacpp", "m")


def test_scan_log_counts_lines():
    log = (
        "INFO init engine (profile, create kv cache, warmup model) took 137.20 seconds\n"
        "WARNING Failed to reload cubin file /root/.triton/cache/x.cubin\n"
        "WARNING Failed to reload cubin file /root/.triton/cache/y.cubin\n"
        "INFO Graph capturing finished in 11 secs, took 0.5 GiB\n"
    )
    found = cc.scan_log("vllm", log)
    assert found["engine_init_s"] == ["137.20"]
    assert len(found["cubin_reload_failed"]) == 2
    assert found["graph_capture_s"] == ["11"]
    assert found["aot_loaded"] == []


def test_labs_import_warns_once_and_is_not_exported():
    code = (
        "import warnings; warnings.simplefilter('always')\n"
        "import saturate.labs.compile_cache, saturate.labs\n"
        "import saturate; assert 'compile_cache' not in saturate.__all__"
    )
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert run.returncode == 0, run.stderr
    assert run.stderr.count("FutureWarning") == 1
    assert "not part of the contract" in run.stderr
