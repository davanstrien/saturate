"""saturate's default log output (saturate/_logs.py): `[pump]` lines on stderr while nobody else
has configured logging, and nothing extra once someone has."""

import io
import logging
import subprocess
import sys
import threading

from saturate import _logs

log = logging.getLogger("saturate.test")


def unconfigured(monkeypatch) -> logging.Logger:
    """No handler on the root logger and only the fallback on `saturate`: a script that never
    touches logging. Restored afterwards, with pytest's own handlers. Called in the test body:
    pytest adds its capture handlers to the root logger after fixtures are set up."""
    package = logging.getLogger("saturate")
    monkeypatch.setattr(logging.getLogger(), "handlers", [])
    fallback = [handler for handler in package.handlers if isinstance(handler, _logs._Fallback)]
    assert len(fallback) == 1, package.handlers
    monkeypatch.setattr(package, "handlers", fallback)
    return package


def test_an_unconfigured_run_shows_info_on_the_stderr_of_the_moment(monkeypatch):
    unconfigured(monkeypatch)
    first, second = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stderr", first)
    log.info("resume: skipped 3 rows")
    monkeypatch.setattr(sys, "stderr", second)  # replaced after the handler was made
    log.info("done: 5 ok")
    assert (first.getvalue(), second.getvalue()) == ("[pump] resume: skipped 3 rows\n", "[pump] done: 5 ok\n")


def test_logging_configured_after_import_gets_each_record_once(monkeypatch, capsys):
    unconfigured(monkeypatch)
    mine = io.StringIO()
    logging.getLogger().addHandler(logging.StreamHandler(mine))  # the caller configures logging
    log.warning("circuit OPEN")
    assert mine.getvalue() == "circuit OPEN\n"
    assert capsys.readouterr().err == ""  # the fallback stays quiet: no second copy


def test_a_level_the_caller_sets_on_saturate_is_kept(monkeypatch, capsys):
    package = unconfigured(monkeypatch)
    monkeypatch.setattr(package, "level", package.level)  # restored afterwards
    package.setLevel(logging.WARNING)
    log.info("done: 5 ok")
    log.warning("circuit OPEN")
    _logs.install()  # importing again changes nothing
    log.info("done: 5 ok")
    assert capsys.readouterr().err == "[pump] circuit OPEN\n"
    assert package.level == logging.WARNING


def test_a_handler_the_caller_put_on_saturate_silences_the_fallback(monkeypatch, capsys):
    package = unconfigured(monkeypatch)
    mine = io.StringIO()
    package.addHandler(logging.StreamHandler(mine))
    log.warning("circuit OPEN")
    assert mine.getvalue() == "circuit OPEN\n" and capsys.readouterr().err == ""


def test_records_from_many_threads_are_printed_once_each(monkeypatch, capsys):
    unconfigured(monkeypatch)
    start = threading.Barrier(8)

    def work(n):
        start.wait()
        log.warning(f"write failed {n}")

    threads = [threading.Thread(target=work, args=(n,)) for n in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    lines = capsys.readouterr().err.splitlines()
    assert sorted(lines) == sorted(f"[pump] write failed {n}" for n in range(8))


def run_script(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)


def test_a_script_that_never_configures_logging_sees_pump_lines():
    """In a fresh process, as a user's script runs: standalone helpers print too."""
    done = run_script("import logging, saturate; logging.getLogger('saturate.engine').info('engine ready')")
    assert (done.stdout, done.stderr) == ("", "[pump] engine ready\n")


def test_a_script_that_configures_logging_at_warning_hides_progress():
    done = run_script("import logging, saturate\n"
                      "logging.basicConfig(level=logging.WARNING, format='%(name)s %(message)s')\n"
                      "logging.getLogger('saturate').setLevel(logging.WARNING)\n"
                      "logging.getLogger('saturate.sink').info('resume: reading 1001 manifest files')\n"
                      "logging.getLogger('saturate.transport').warning('circuit OPEN')")
    assert (done.stdout, done.stderr) == ("", "saturate.transport circuit OPEN\n")
