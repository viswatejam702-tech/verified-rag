#!/usr/bin/env python3
import os
import sys
import subprocess

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
script = os.path.join(root_dir, "eval", "test_pipeline.py")
sys.exit(subprocess.call([sys.executable, script] + sys.argv[1:], cwd=root_dir))

