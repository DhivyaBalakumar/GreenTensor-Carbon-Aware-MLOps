"""Runs the stress test with a hard timeout per section."""
import subprocess, sys, time

try:
    result = subprocess.run(
        [sys.executable, "stress_test_greentensor.py"],
        capture_output=True, text=True, timeout=180,
        cwd=".", encoding="utf-8", errors="replace"
    )
    output = result.stdout + result.stderr
    print(output[-10000:])
    print(f"\nReturn code: {result.returncode}")
except subprocess.TimeoutExpired as e:
    out = e.stdout or ""
    err = e.stderr or ""
    if isinstance(out, bytes): out = out.decode("utf-8", errors="replace")
    if isinstance(err, bytes): err = err.decode("utf-8", errors="replace")
    combined = out + err
    print("=== TIMED OUT after 90s. Last output: ===")
    print(combined[-5000:])
