import pytest

import magna.cli.build as build_mod


def test_install_raylib_skips_when_headers_present(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(build_mod, "is_raylib_installed", lambda: True)
    monkeypatch.setattr(build_mod, "_run", lambda *args, **kwargs: calls.append(args))

    build_mod.install_raylib(verbose=True)

    assert calls == []
    assert "already present" in capsys.readouterr().out


def test_install_raylib_builds_from_source(monkeypatch):
    calls = []
    installed = [False, True]
    monkeypatch.setattr(build_mod, "is_raylib_installed", lambda: installed.pop(0))

    def fake_run(cmd, verbose, cwd=None):
        calls.append((cmd, cwd))

    monkeypatch.setattr(build_mod, "_run", fake_run)

    build_mod.install_raylib()

    commands = [cmd for cmd, _ in calls]
    assert commands[0] == [
        "sudo",
        "apt",
        "install",
        "-y",
        *build_mod.RAYLIB_APT_PACKAGES,
    ]
    assert commands[1][:3] == ["git", "clone", "https://github.com/raysan5/raylib.git"]
    assert commands[1][3].endswith("/raylib")
    assert commands[2:] == [
        ["cmake", "-DBUILD_SHARED_LIBS=ON", ".."],
        ["make"],
        ["sudo", "make", "install"],
        ["sudo", "ldconfig"],
    ]
    assert all(cwd.name == "build" for _, cwd in calls[2:5])


def test_install_raylib_errors_when_header_missing_after_install(monkeypatch):
    installed = [False, False]
    monkeypatch.setattr(build_mod, "is_raylib_installed", lambda: installed.pop(0))
    monkeypatch.setattr(build_mod, "_run", lambda *args, **kwargs: None)

    with pytest.raises(SystemExit, match="Failed to find raylib"):
        build_mod.install_raylib()


def test_install_raylib_only_cli_exits_without_building(monkeypatch):
    calls = []
    monkeypatch.setattr(
        build_mod, "install_raylib", lambda verbose=False: calls.append(verbose)
    )

    assert build_mod.main(["--install-raylib-only", "--verbose"]) == 0
    assert calls == [True]
