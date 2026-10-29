#!/usr/bin/env python3
"""
Root launcher for verify_keys.py
"""

import os
import sys
import subprocess

script_dir = os.path.dirname(os.path.abspath(__file__))
verify_script = os.path.join(script_dir, "eval", "verify_keys.py")

sys.exit(subprocess.call([sys.executable, verify_script] + sys.argv[1:], cwd=script_dir))
