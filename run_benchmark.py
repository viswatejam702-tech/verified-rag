#!/usr/bin/env python3
"""
Root runner for Evaluation Benchmark
"""

import os
import sys
import subprocess

script_dir = os.path.dirname(os.path.abspath(__file__))
bench_script = os.path.join(script_dir, "eval", "benchmark.py")

sys.exit(subprocess.call([sys.executable, bench_script] + sys.argv[1:], cwd=script_dir))
