"""saturate's default log output (saturate/_logs.py): `[pump]` lines on stderr while nobody else
has configured logging, and out of the way once someone has."""

import logging

from saturate import _logs

log = logging.getLogger("saturate.test")


def unconfigured(monkeypatch) -> logging.Logger:
    """No handler on the root logger or on `saturate`, and `saturate` at NOTSET: a script that
    never touches logging. Restored afterwards, with pytest's own handlers. Called in the test
    body: pytest adds its capture handlers to the root logger after fixtures are set up."""
    package = logging.getLogger("saturate")
    monkeypatch.setattr(logging.getLogger(), "handlers", [])
    monkeypatch.setattr(package, "handlers", [])
    monkeypatch.setattr(package, "level", logging.NOTSET)
    return package


def test_an_unconfigured_run_shows_info_on_the_current_stderr(monkeypatch, capsys):
    unconfigured(monkeypatch)
    _logs.default_output()
    log.info("done: 5 ok")
    assert capsys.readouterr().err == "[pump] done: 5 ok\n"


def test_the_default_output_is_added_once(monkeypatch, capsys):
    unconfigured(monkeypatch)
    _logs.default_output()
    _logs.default_output()
    log.warning("circuit OPEN")
    assert capsys.readouterr().err == "[pump] circuit OPEN\n"


def test_configuring_logging_later_removes_the_default_output(monkeypatch, capsys):
    package = unconfigured(monkeypatch)
    _logs.default_output()
    logging.getLogger().addHandler(logging.NullHandler())  # the caller configures logging
    _logs.default_output()
    assert package.handlers == [] and package.level == logging.NOTSET
    log.warning("circuit OPEN")
    assert capsys.readouterr().err == ""


def test_a_handler_the_caller_put_on_saturate_is_left_alone(monkeypatch, capsys):
    package = unconfigured(monkeypatch)
    mine = logging.NullHandler()
    package.addHandler(mine)
    package.setLevel(logging.ERROR)
    _logs.default_output()
    assert package.handlers == [mine] and package.level == logging.ERROR
    log.warning("circuit OPEN")
    assert capsys.readouterr().err == ""
