"""saturate.labs.compile_cache on fake cache folders, with a local folder as the bucket."""

import json
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
    for name in (
        "VLLM_CACHE_ROOT",
        "SGLANG_CACHE_DIR",
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_XET_CACHE",
        "HF_ASSETS_CACHE",
        "HF_TOKEN_PATH",
        "TRITON_CACHE_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    return home


@pytest.fixture
def profile(home):
    return cc.Profile(
        groups={
            "engine": ("~/.cache/engine",),
            "triton": ("~/.triton", "$TRITON_CACHE_DIR"),
            "ptx": ("~/.nv",),
        },
        never=("*.lock", "locks", "runs", "assets/*", "gpu_p2p_*", ".cc-*"),
        env={"TRITON_CACHE_DIR": "~/.triton/cache"},
    )


def write(path, text="x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def members(archive):
    with tarfile.open(archive) as tar:
        return sorted(m.name for m in tar.getmembers() if m.isfile())


def wipe(*folders):
    """A fresh machine: the cache folders are gone."""
    for folder in folders:
        for path in sorted(folder.rglob("*"), reverse=True):
            path.unlink() if path.is_file() or path.is_symlink() else path.rmdir()


def test_cold_save_then_warm_restore_saves_nothing(tmp_path, home, profile):
    bucket = tmp_path / "bucket"
    with cc.compile_cache(str(bucket), profile, "vllm/m/gpu/v1") as cache:
        assert cache.report.restore == "missing"
        write(home / ".cache/engine/hash1/graph.py", "graph")
        kernel = write(home / ".triton/cache/abc/kernel.cubin", "cubin")
        os.utime(kernel, (1_000_000, 1_000_000))
    archive = bucket / "vllm/m/gpu/v1.tar"
    assert cache.report.saved and cache.report.new_files == 2
    assert members(archive) == ["home/.cache/engine/hash1/graph.py", "home/.triton/cache/abc/kernel.cubin"]
    assert cache.report.groups == {
        "engine": {"restored": 0, "new": 1, "changed": 0},
        "triton": {"restored": 0, "new": 1, "changed": 0},
    }

    wipe(home / ".cache", home / ".triton")
    with cc.compile_cache(str(bucket), profile, "vllm/m/gpu/v1") as cache:
        assert (home / ".cache/engine/hash1/graph.py").read_text() == "graph"
        assert kernel.read_text() == "cubin"
        assert kernel.stat().st_mtime == 1_000_000  # kept: build tools compare mtimes
    report = cache.report
    assert report.restore == "restored" and report.restored_files == 2
    assert report.groups["triton"]["restored"] == 1
    assert not report.saved and report.new_files == 0
    assert json.loads(report.to_json())["restore"] == "restored"


def test_without_a_with_block(tmp_path, home, profile):
    cache = cc.compile_cache(str(tmp_path / "bucket"), profile, "k")
    cache.restore()
    assert os.environ["TRITON_CACHE_DIR"] == str(home / ".triton/cache")
    write(home / ".cache/engine/a.py")
    cache.save()
    cache.close()
    assert cache.report.saved and "TRITON_CACHE_DIR" not in os.environ
    with pytest.raises(RuntimeError, match="requires an open cache"):
        cache.save()


def test_new_or_changed_file_saves_the_union(tmp_path, home, profile):
    bucket = tmp_path / "bucket"
    with cc.compile_cache(str(bucket), profile, "k"):
        write(home / ".cache/engine/a.py", "a")
        write(home / ".cache/engine/b.json", "{}")
    with cc.compile_cache(str(bucket), profile, "k") as cache:
        write(home / ".cache/engine/c.py", "c")  # new
        write(home / ".cache/engine/b.json", '{"tuned": 1}')  # changed size
    assert cache.report.saved and cache.report.new_files == 1 and cache.report.changed_files == 1
    assert members(bucket / "k.tar") == [
        "home/.cache/engine/a.py",
        "home/.cache/engine/b.json",
        "home/.cache/engine/c.py",
    ]


def test_rewritten_file_of_the_same_size_saves_nothing(tmp_path, home, profile):
    bucket = tmp_path / "bucket"
    with cc.compile_cache(str(bucket), profile, "k"):
        write(home / ".cache/engine/autotune.json", "1111")
    with cc.compile_cache(str(bucket), profile, "k") as cache:
        write(home / ".cache/engine/autotune.json", "2222")  # rewritten every run
    assert not cache.report.saved


def test_skip_leaves_a_group_out(tmp_path, home, profile):
    bucket = tmp_path / "bucket"
    with cc.compile_cache(str(bucket), profile, "k", skip={"ptx"}) as cache:
        write(home / ".cache/engine/a.py")
        write(home / ".nv/ComputeCache/big.bin")
    assert members(bucket / "k.tar") == ["home/.cache/engine/a.py"]
    assert cache.report.skipped_groups == ["ptx"]
    with pytest.raises(ValueError, match="skip= names from"):
        cc.compile_cache(str(bucket), profile, "k", skip={"ptxx"})


def test_never_files_and_the_hf_cache_stay_out(tmp_path, home, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(home / ".cache/engine/hf"))  # inside a listed folder
    profile = cc.Profile(groups={"all": ("~/.cache",)}, never=cc.SGLANG.never)
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
        write(home / ".cache/huggingface/token", "hf_secret")  # the default home stays guarded too
        os.symlink(home / ".cache/engine/jit/kernel.so", home / ".cache/engine/link.so")
    assert members(bucket / "k.tar") == ["home/.cache/engine/jit/kernel.so"]


def test_restore_refuses_members_outside_the_folders(tmp_path, home, profile):
    bucket = tmp_path / "bucket"
    bucket.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    os.makedirs(home / ".cache/engine")
    os.symlink(outside, home / ".cache/engine/escape")  # a link inside the folder, pointing out
    payload = write(tmp_path / "payload", "evil")
    with tarfile.open(bucket / "k.tar", "w") as tar:
        tar.add(payload, arcname="home/.cache/engine/ok.py")
        tar.add(payload, arcname="home/.bashrc")
        tar.add(payload, arcname="home/.cache/engine/../../.profile")
        tar.add(payload, arcname="home/.cache/engine/assets/big.bin")
        tar.add(payload, arcname="home/.cache/engine/escape/planted.py")
        tar.add(payload, arcname="home/.cache/engine/escape/deeper/planted.py")
        tar.add(payload, arcname="root/etc/evil")
        tar.add(payload, arcname="elsewhere/x")
    with cc.compile_cache(str(bucket), profile, "k", save=False) as cache:
        pass
    assert cache.report.restored_files == 1 and cache.report.refused_files == 7
    assert (home / ".cache/engine/ok.py").exists()
    assert not (home / ".bashrc").exists() and not (home / ".profile").exists()
    assert not (home / ".cache/engine/assets").exists()
    assert list(outside.iterdir()) == []


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
    with cc.compile_cache(str(bucket), profile, "k") as cache:
        assert os.environ["TRITON_CACHE_DIR"] == str(mine)
        write(mine / "k.cubin")
    assert cache.report.env == {}
    assert os.environ["TRITON_CACHE_DIR"] == str(mine)
    assert members(bucket / "k.tar") == ["root" + str(mine / "k.cubin")]
    assert cache.report.groups["triton"]["new"] == 1


def test_unreadable_archive_is_recorded_and_never_replaced(tmp_path, home, profile, caplog):
    bucket = tmp_path / "bucket"
    write(bucket / "k.tar", "not a tar")
    with cc.compile_cache(str(bucket), profile, "k") as cache:
        write(home / ".cache/engine/a.py")
    assert cache.report.restore == "failed" and "ReadError" in (cache.report.restore_error or "")
    assert not cache.report.saved and (bucket / "k.tar").read_text() == "not a tar"
    assert "not saving" in caplog.text


def test_save_inside_the_block_survives_a_later_exception(tmp_path, home, profile):
    bucket = tmp_path / "bucket"
    with pytest.raises(RuntimeError), cc.compile_cache(str(bucket), profile, "k") as cache:
        write(home / ".cache/engine/boot.py")  # compiled while the engine booted
        cache.save()
        write(home / ".cache/engine/later.py")
        raise RuntimeError("pump failed")
    assert members(bucket / "k.tar") == ["home/.cache/engine/boot.py"]
    assert cache.report.saved and cache.report.new_files == 1


def test_exit_save_compares_with_the_save_inside_the_block(tmp_path, home, profile):
    bucket = tmp_path / "bucket"
    with cc.compile_cache(str(bucket), profile, "k") as cache:
        write(home / ".cache/engine/boot.py")
        cache.save()
    assert cache.report.new_files == 1  # the exit found nothing new: one save
    with cc.compile_cache(str(bucket), profile, "k") as cache:
        cache.save()
        write(home / ".cache/engine/inference.py")  # compiled while serving
    assert cache.report.new_files == 1 and not cache.report.replaced_other_writer
    assert len(members(bucket / "k.tar")) == 2


def test_another_writer_since_the_restore_is_reported(tmp_path, home, profile, caplog):
    bucket = tmp_path / "bucket"
    first = cc.compile_cache(str(bucket), profile, "k")
    first.restore()  # finds no archive
    with cc.compile_cache(str(bucket), profile, "k"):  # another run saves meanwhile
        write(home / ".cache/engine/theirs.py")
    write(home / ".cache/engine/mine.py")
    first.save()
    first.close()
    assert first.report.replaced_other_writer and "another run saved" in caplog.text


def test_restore_keeps_files_that_exist(tmp_path, home, profile):
    bucket = tmp_path / "bucket"
    with cc.compile_cache(str(bucket), profile, "k"):
        write(home / ".cache/engine/shipped.so", "from the archive")
    write(home / ".cache/engine/shipped.so", "from the image")
    with cc.compile_cache(str(bucket), profile, "k", save=False) as cache:
        pass
    assert (home / ".cache/engine/shipped.so").read_text() == "from the image"
    assert cache.report.existing_files == 1 and cache.report.restored_files == 0


def test_truncated_archive_leaves_no_partial_file(tmp_path, home, profile):
    bucket = tmp_path / "bucket"
    with cc.compile_cache(str(bucket), profile, "k"):
        write(home / ".cache/engine/a.so", "a" * 2000)
        write(home / ".cache/engine/b.so", "b" * 20000)
    archive = bucket / "k.tar"
    archive.write_bytes(archive.read_bytes()[:8000])  # ends inside b.so's data
    wipe(home / ".cache/engine")
    with cc.compile_cache(str(bucket), profile, "k", save=False) as cache:
        pass
    assert sorted(p.name for p in (home / ".cache/engine").iterdir()) == ["a.so"]
    assert cache.report.restore == "failed"


def test_archive_moves_to_a_different_home(tmp_path, home, profile, monkeypatch):
    bucket = tmp_path / "bucket"
    with cc.compile_cache(str(bucket), profile, "k"):
        write(home / ".cache/engine/a.py", "a")
    other = tmp_path / "other-home"
    other.mkdir()
    monkeypatch.setenv("HOME", str(other))
    with cc.compile_cache(str(bucket), profile, "k", save=False) as cache:
        pass
    assert (other / ".cache/engine/a.py").read_text() == "a" and cache.report.restored_files == 1


def test_failed_upload_is_recorded_and_the_block_does_not_raise(tmp_path, home, profile, monkeypatch):
    def broken(fs, local, path):
        raise OSError("bucket unreachable")

    monkeypatch.setattr(cc, "_put", broken)
    with cc.compile_cache(str(tmp_path / "bucket"), profile, "k") as cache:
        write(home / ".cache/engine/a.py")
    assert not cache.report.saved and "bucket unreachable" in (cache.report.save_error or "")


def test_a_file_that_fails_to_pack_fails_the_whole_save(tmp_path, home, profile, monkeypatch):
    bucket = tmp_path / "bucket"
    with cc.compile_cache(str(bucket), profile, "k"):
        write(home / ".cache/engine/a.py")
    add = tarfile.TarFile.add

    def shrinking(self, name, *args, **kwargs):
        if name.endswith("b.py"):
            raise OSError("unexpected end of data")  # what tarfile raises when a file shrinks
        return add(self, name, *args, **kwargs)

    monkeypatch.setattr(tarfile.TarFile, "add", shrinking)
    with cc.compile_cache(str(bucket), profile, "k") as cache:
        write(home / ".cache/engine/b.py")
        write(home / ".cache/engine/c.py")
    assert not cache.report.saved and "unexpected end" in (cache.report.save_error or "")
    assert members(bucket / "k.tar") == ["home/.cache/engine/a.py"]  # the old archive, whole


def test_engine_profiles_and_bad_arguments(home, monkeypatch, tmp_path):
    monkeypatch.setenv("SGLANG_CACHE_DIR", str(tmp_path / "sg"))
    folders = cc._folders(cc.SGLANG)
    assert folders[str(tmp_path / "sg/jit")] == "jit" and str(tmp_path / "sg") not in folders
    assert str(tmp_path / "sg/nv") not in cc._folders(cc.SGLANG, skip={"ptx"})
    with pytest.raises(ValueError, match="engine= from"):
        cc.compile_cache(str(tmp_path), "tgi", "k")
    with pytest.raises(ValueError, match="relative key"):
        cc.compile_cache(str(tmp_path), "vllm", "../k")


@pytest.mark.internals
def test_vllm_profile_pins_triton_and_names_subfolders(home, monkeypatch, tmp_path):
    (tmp_path / "scratch").mkdir()  # gettempdir() skips a TMPDIR that does not exist
    monkeypatch.setenv("TMPDIR", str(tmp_path / "scratch"))
    monkeypatch.setattr(cc.tempfile, "tempdir", None)  # gettempdir() reads TMPDIR again
    monkeypatch.setenv("TRITON_CACHE_DIR", str(home / ".triton/cache"))
    folders = cc._folders(cc.VLLM)
    assert folders[str(home / ".cache/vllm/torch_compile_cache")] == "compile"
    assert str(home / ".cache/vllm") not in folders  # its assets folder stays out
    assert str(home / ".triton/cache") not in folders  # inside ~/.triton, listed once
    assert str(tmp_path / "scratch" / f"torchinductor_{cc._user()}") in folders
    assert cc.VLLM.env == {"TRITON_CACHE_DIR": "~/.triton/cache"}


@pytest.mark.internals
def test_local_upload_leaves_no_partial_on_failure(tmp_path, monkeypatch):
    import fsspec

    fs = fsspec.filesystem("file")
    local = write(tmp_path / "save.tar", "tar")
    target = tmp_path / "bucket" / "k.tar"
    target.parent.mkdir()

    def broken_mv(*args, **kwargs):
        raise OSError("rename failed")

    monkeypatch.setattr(fs, "mv", broken_mv)  # fsspec shares one instance: patch it, then put it back
    with pytest.raises(OSError):
        cc._put(fs, str(local), str(target))
    assert list(target.parent.iterdir()) == []


@pytest.mark.internals
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


@pytest.mark.internals
def test_gpu_name_capability_and_count(monkeypatch):
    class Done:
        stdout = "NVIDIA H200, 9.0\nNVIDIA H200, 9.0\n"

    monkeypatch.setattr(cc.subprocess, "run", lambda *a, **k: Done())
    assert cc._gpu() == "NVIDIA-H200-sm90-x2"
    Done.stdout = "garbage\n"
    with pytest.raises(ValueError, match="requires gpu="):
        cc._gpu()


def test_default_key(monkeypatch):
    key = cc.default_key(
        "sglang",
        "baidu/Unlimited-OCR-2",
        version="0.5.20.dev10+g8d62f5fb9c",
        image="lmsysorg/sglang:v0.5.20-cu130",
        gpu="NVIDIA-RTX-PRO-6000-sm120-x1",
        extra=["tp1"],
    )
    assert key == (
        "sglang/baidu--Unlimited-OCR-2/NVIDIA-RTX-PRO-6000-sm120-x1/"
        "sglang-0.5.20.dev10-g8d62f5fb9c/lmsysorg--sglang-v0.5.20-cu130/tp1"
    )


def test_default_key_without_a_version_raises(monkeypatch):
    monkeypatch.setattr("saturate.engine.python_with", lambda module: "/nonexistent/python3")
    with pytest.raises(ValueError, match="requires version="):
        cc.default_key("vllm", "m", gpu="gpu")
    with pytest.raises(ValueError, match="requires version="):
        cc.default_key("sglang", "m", gpu="gpu")


def test_labs_import_warns_once_and_is_not_exported():
    code = (
        "import warnings; warnings.simplefilter('always')\n"
        "import saturate.labs.compile_cache, saturate.labs\n"
        "import saturate; assert 'compile_cache' not in saturate.__all__"
    )
    env = {k: v for k, v in os.environ.items() if k != "PYTHONWARNINGS"}
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False, env=env)
    assert run.returncode == 0, run.stderr
    assert run.stderr.count("saturate.labs holds lab helpers") == 1
