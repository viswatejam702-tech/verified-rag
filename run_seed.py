#!/usr/bin/env python3
"""
Root runner for Seed Data Ingestion
"""

import os
import sys
import subprocess

script_dir = os.path.dirname(os.path.abspath(__file__))
seed_script = os.path.join(script_dir, "seed-data", "load_seed.py")

sys.exit(subprocess.call([sys.executable, seed_script] + sys.argv[1:], cwd=script_dir))
