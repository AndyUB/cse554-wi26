"""Deprecated wrapper.

Use profile_vllm.sh instead.
"""

import pathlib
import subprocess


if __name__ == "__main__":
    script = pathlib.Path(__file__).with_suffix(".sh")
    raise SystemExit(subprocess.call(["bash", str(script)]))
