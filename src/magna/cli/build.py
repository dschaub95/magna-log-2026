"""Build the local C binding for the what env."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sysconfig
import tempfile
from pathlib import Path

RAYLIB_APT_PACKAGES = [
    "build-essential",
    "git",
    "cmake",
    "libasound2-dev",
    "libx11-dev",
    "libxrandr-dev",
    "libxi-dev",
    "libgl1-mesa-dev",
    "libglu1-mesa-dev",
    "libxcursor-dev",
    "libxinerama-dev",
    "libwayland-dev",
    "libxkbcommon-dev",
]

QHULL_HEADER = Path("/usr/include/libqhull_r/libqhull_r.h")


def _run(cmd: list[str], verbose: bool, cwd: Path | None = None) -> None:
    if verbose:
        prefix = f"(cd {shlex.quote(str(cwd))}) " if cwd else ""
        print("+", prefix + " ".join(shlex.quote(part) for part in cmd))
    subprocess.run(cmd, check=True, cwd=cwd)


def install_qhull(verbose: bool = False) -> None:
    if QHULL_HEADER.exists():
        return
    _run(["sudo", "apt", "install", "-y", "libqhull-dev"], verbose)
    if not QHULL_HEADER.exists():
        raise SystemExit("Install libqhull-dev (or qhull-devel) before building.")


def _split_flags(flags: str) -> list[str]:
    if not flags:
        return []
    return shlex.split(flags)


def _opt_name(prefix: str, name: str) -> str:
    base = f"{prefix}{name}" if prefix else name
    return base.replace("_", "-")


def _dest_name(prefix: str, name: str) -> str:
    return f"{prefix}{name}".replace("-", "_")


def is_raylib_installed() -> bool:
    return any(
        path.exists()
        for path in (Path("/usr/include/raylib.h"), Path("/usr/local/include/raylib.h"))
    )


def install_raylib(verbose: bool = False) -> None:
    if is_raylib_installed():
        print("raylib headers already present.")
        return

    print("Installing raylib build dependencies...")
    _run(["sudo", "apt", "install", "-y", *RAYLIB_APT_PACKAGES], verbose)

    with tempfile.TemporaryDirectory(prefix="magna-raylib-") as tmp:
        source_dir = Path(tmp) / "raylib"
        build_dir = source_dir / "build"
        _run(
            ["git", "clone", "https://github.com/raysan5/raylib.git", str(source_dir)],
            verbose,
        )
        build_dir.mkdir(parents=True)
        _run(["cmake", "-DBUILD_SHARED_LIBS=ON", ".."], verbose, cwd=build_dir)
        _run(["make"], verbose, cwd=build_dir)
        _run(["sudo", "make", "install"], verbose, cwd=build_dir)

    _run(["sudo", "ldconfig"], verbose)
    if not is_raylib_installed():
        raise SystemExit("Failed to find raylib after source installation.")


def add_build_args(parser: argparse.ArgumentParser, prefix: str = "") -> None:
    parser.add_argument(
        f"--{_opt_name(prefix, 'output-dir')}",
        dest=_dest_name(prefix, "output_dir"),
        type=Path,
        default=Path("src/magna/env"),
        help="Where to place the built binding shared library",
    )
    parser.add_argument(
        f"--{_opt_name(prefix, 'build-dir')}",
        dest=_dest_name(prefix, "build_dir"),
        type=Path,
        default=Path("build/env"),
        help="Directory for intermediate build artifacts",
    )
    parser.add_argument(
        f"--{_opt_name(prefix, 'verbose')}",
        dest=_dest_name(prefix, "verbose"),
        action="store_true",
        help="Print compiler commands",
    )


def build_argv_from_namespace(args: argparse.Namespace, prefix: str = "") -> list[str]:
    build_argv: list[str] = []
    if output_dir := getattr(args, _dest_name(prefix, "output_dir")):
        build_argv.extend(["--output-dir", str(output_dir)])
    if build_dir := getattr(args, _dest_name(prefix, "build_dir")):
        build_argv.extend(["--build-dir", str(build_dir)])
    if getattr(args, _dest_name(prefix, "verbose")):
        build_argv.append("--verbose")
    return build_argv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build magna env C binding")
    add_build_args(parser)
    parser.add_argument(
        "--install-raylib-only",
        action="store_true",
        help="Install raylib dev headers and exit without building the binding",
    )
    args = parser.parse_args(argv)

    if args.install_raylib_only:
        install_raylib(args.verbose)
        return 0

    root = Path(__file__).resolve().parents[3]
    env_dir = root / "src" / "magna" / "env"
    output_dir = (root / args.output_dir).resolve()
    build_dir = (root / args.build_dir).resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import numpy
    except ImportError as exc:
        raise SystemExit("NumPy is required to build the env binding.") from exc

    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    target = output_dir / f"binding{ext_suffix}"
    obj = build_dir / "binding.o"
    utils_obj = build_dir / "utils.o"

    python_include = sysconfig.get_paths()["include"]
    include_dirs = [python_include, numpy.get_include(), str(env_dir)]
    include_flags = [f"-I{path}" for path in include_dirs]

    cflags = _split_flags(sysconfig.get_config_var("CFLAGS") or "")
    env_cflags = _split_flags(os.environ.get("CFLAGS", ""))

    cxxflags = _split_flags(
        sysconfig.get_config_var("CXXFLAGS") or sysconfig.get_config_var("CFLAGS") or ""
    )
    env_cxxflags = _split_flags(os.environ.get("CXXFLAGS", ""))

    ldflags = _split_flags(sysconfig.get_config_var("LDFLAGS") or "")
    ldflags += _split_flags(os.environ.get("LDFLAGS", ""))

    if not is_raylib_installed():
        print("raylib not found. Installing raylib dev headers...")
        install_raylib(args.verbose)

    if not QHULL_HEADER.exists():
        print("qhull not found. Installing libqhull-dev...")
        install_qhull(args.verbose)

    cc = os.environ.get("CC") or sysconfig.get_config_var("CC") or "cc"
    if isinstance(cc, str):
        cc = shlex.split(cc)

    cxx = os.environ.get("CXX") or sysconfig.get_config_var("CXX") or "g++"
    if isinstance(cxx, str):
        cxx = shlex.split(cxx)

    ldshared = sysconfig.get_config_var("LDSHARED")
    if ldshared:
        ld_cmd = shlex.split(ldshared)
        default_cc = _split_flags(sysconfig.get_config_var("CC") or "")
        prefix_len = (
            len(default_cc)
            if default_cc and ld_cmd[: len(default_cc)] == default_cc
            else 1
        )
        ld_cmd = cxx + ld_cmd[prefix_len:]
    else:
        ld_cmd = cxx + ["-shared"]

    _run(
        [
            *cc,
            *cflags,
            "-O3",
            "-fPIC",
            "-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION",
            *env_cflags,
            *include_flags,
            "-c",
            str(env_dir / "binding.c"),
            "-o",
            str(obj),
        ],
        args.verbose,
    )

    _run(
        [
            *cxx,
            *cxxflags,
            "-O3",
            "-fPIC",
            "-std=c++17",
            "-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION",
            *env_cxxflags,
            *include_flags,
            "-c",
            str(env_dir / "utils.cpp"),
            "-o",
            str(utils_obj),
        ],
        args.verbose,
    )

    _run(
        [
            *ld_cmd,
            str(obj),
            str(utils_obj),
            "-o",
            str(target),
            *ldflags,
            "-lraylib",
            "-lqhull_r",
        ],
        args.verbose,
    )

    print(f"Built {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
