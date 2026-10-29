#!/usr/bin/env python3
import os
import sys
import subprocess

script_dir = os.path.dirname(os.path.abspath(__file__))
script = os.path.join(script_dir, "eval", "test_supabase_connection.py")
sys.exit(subprocess.call([sys.executable, script] + sys.argv[1:], cwd=script_dir))
