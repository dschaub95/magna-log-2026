import subprocess
import sys


def main():
    argv = sys.argv[1:]
    use_sweep = "--sweep" in argv
    filtered_argv = [arg for arg in argv if arg not in ("--sweep", "--train")]
    module = "magna.cli.sweep" if use_sweep else "magna.cli.train"
    script_name = "sweep" if use_sweep else "train"

    print(
        f"""======================================================================
🐛 DEBUG MODE
======================================================================
Starting debugpy server for: {script_name}
Listening on port: 5678

📋 NEXT STEPS:
1. Open the Run and Debug panel in Cursor (Ctrl+Shift+D / Cmd+Shift+D)
2. Select 'Python: Attach to debugpy' from the dropdown
3. Click the green play button (▶) or press F5
4. The debugger will attach and execution will begin

💡 TIP: Set breakpoints in your code before attaching the debugger
======================================================================
"""
    )

    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "debugpy",
        "--listen",
        "5678",
        "--wait-for-client",
        "-m",
        module,
    ] + filtered_argv

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n\n⚠️  Debug session interrupted by user")
        sys.exit(0)
    except subprocess.CalledProcessError as e:
        print(f"\n\n❌ Error running debug command: {e}", file=sys.stderr)
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
