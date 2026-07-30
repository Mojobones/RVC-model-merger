"""
Launches the frozen build and confirms it starts and stays running, rather than
just confirming PyInstaller exited 0. A build can succeed and still produce a
binary that crashes on launch - missing Tcl/Tk data files and a broken
tkinterdnd2 native library bundling are exactly the kind of thing that only
shows up when something actually tries to run.
"""
import subprocess
import sys
import time

STARTUP_WAIT_SECONDS = 5


def main():
    if len(sys.argv) != 2:
        print("usage: smoke_test.py <path-to-binary>")
        return 1

    binary = sys.argv[1]
    proc = subprocess.Popen(
        [binary],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    time.sleep(STARTUP_WAIT_SECONDS)
    exit_code = proc.poll()

    if exit_code is not None:
        output = proc.stdout.read() if proc.stdout else ""
        print(f"FAILED: process exited immediately with code {exit_code}")
        print("--- output ---")
        print(output)
        return 1

    print(f"OK: still running after {STARTUP_WAIT_SECONDS}s, terminating.")
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
