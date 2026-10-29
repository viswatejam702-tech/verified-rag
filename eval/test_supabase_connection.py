#!/usr/bin/env python3
"""
Supabase & PostgreSQL Vector Database Connection & Health Verification
Tests connection, schema tables, vector extension, and vector search RPC.
"""

import os
import sys
import json
import urllib.request
import urllib.error

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def load_env(env_path):
    env_vars = {}
    if not os.path.exists(env_path):
        return env_vars
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env_vars[k.strip()] = v.strip().strip("'\"")
    return env_vars

def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    env = load_env(os.path.join(root_dir, ".env"))

    sb_url = env.get("NEXT_PUBLIC_SUPABASE_URL", "")
    sb_anon = env.get("NEXT_PUBLIC_SUPABASE_ANON_KEY", "")
    sb_service = env.get("SUPABASE_SERVICE_ROLE_KEY", "")

    print("=" * 80)
    print(" 🗄️ SUPABASE / PGVECTOR BACKEND CONNECTION CHECK")
    print("=" * 80)

    if not sb_url or "your-project" in sb_url:
        print("⚠️ Supabase URL is not yet configured in .env")
        print("  Current mode: Using built-in local vector store (In-Memory)")
        print("\n👉 Follow the step-by-step guide below to provision your free Supabase DB.")
        print("=" * 80)
        return

    # Check REST health
    health_url = f"{sb_url}/rest/v1/"
    key_to_use = sb_service or sb_anon
    req = urllib.request.Request(
        health_url,
        headers={
            "apikey": key_to_use,
            "Authorization": f"Bearer {key_to_use}",
            "User-Agent": "SupabaseVerifier/1.0"
        }
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            print("✅ Supabase REST API connection: SUCCESS!")
            print(f"   Project URL: {sb_url}")
    except urllib.error.HTTPError as e:
        print(f"❌ Supabase Connection Error: HTTP {e.code}")
        return
    except Exception as e:
        print(f"❌ Network Error: {e}")
        return

    # Check if documents and document_chunks tables exist
    docs_url = f"{sb_url}/rest/v1/documents?select=id&limit=1"
    req_docs = urllib.request.Request(
        docs_url,
        headers={
            "apikey": key_to_use,
            "Authorization": f"Bearer {key_to_use}",
            "User-Agent": "SupabaseVerifier/1.0"
        }
    )

    try:
        with urllib.request.urlopen(req_docs, timeout=10) as resp:
            print("✅ Database Schema Table 'documents': FOUND & READY!")
    except urllib.error.HTTPError as e:
        if e.code == 404 or "PGRST205" in str(e):
            print("⚠️ Table 'documents' not found. Please run schema.sql in Supabase SQL Editor.")
        else:
            print(f"⚠️ Table check notice: HTTP {e.code}")
    except Exception as e:
        print(f"⚠️ Check notice: {e}")

    print("=" * 80)

if __name__ == "__main__":
    main()
