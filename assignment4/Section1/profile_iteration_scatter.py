"""Deprecated shim.

Per-iteration scatter profiling has been merged into
`profile_continous_vs_chunked.py`.
"""

import subprocess
import sys


if __name__ == "__main__":
    raise SystemExit(
        subprocess.call(
            [
                sys.executable,
                "profile_continous_vs_chunked.py",
                "--with-scatter",
                "--scatter-json",
                "iteration_times.json",
                "--scatter-png",
                "iteration_times_scatter.png",
            ]
        )
    )
