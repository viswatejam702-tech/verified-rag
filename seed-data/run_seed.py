#!/usr/bin/env python3
import os
import sys
import subprocess

current_dir = os.path.dirname(os.path.abspath(__file__))
script = os.path.join(current_dir, "load_seed.py")
sys.exit(subprocess.call([sys.executable, script] + sys.argv[1:], cwd=current_dir))
