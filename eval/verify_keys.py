#!/usr/bin/env python3
"""
API Keys Live Verification Script
Tests OpenAI, Google Gemini, Anthropic Claude, and Hugging Face keys against official APIs.
Masks all sensitive keys in outputs.
"""

import os
import sys
import json
import urllib.request
import urllib.error

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Load .env manually to avoid extra dependencies
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


def mask_key(k: str) -> str:
    if not k:
        return "<NOT SET>"
    if len(k) <= 8:
        return k[:2] + "..." + k[-2:]
    return k[:6] + "..." + k[-4:]


def test_openai(api_key: str):
    if not api_key:
        return False, "Not configured in .env"
    url = "https://api.openai.com/v1/models"
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {api_key}", "User-Agent": "KeyVerifier/1.0"}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            count = len(data.get("data", []))
            return True, f"Valid! (Successfully accessed {count} models)"
    except urllib.error.HTTPError as e:
        err_b = e.read().decode("utf-8", errors="ignore")
        try:
            err_j = json.loads(err_b)
            return False, f"HTTP {e.code}: {err_j.get('error', {}).get('message', err_b)}"
        except:
            return False, f"HTTP {e.code}: {err_b}"
    except Exception as e:
        return False, str(e)


def test_gemini(api_key: str):
    if not api_key:
        return False, "Not configured in .env"
    # Try listing models
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    req = urllib.request.Request(url, headers={"User-Agent": "KeyVerifier/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = [m.get("name") for m in data.get("models", [])]
            return True, f"Valid! (Successfully accessed Gemini models, count={len(models)})"
    except urllib.error.HTTPError as e:
        err_b = e.read().decode("utf-8", errors="ignore")
        try:
            err_j = json.loads(err_b)
            return False, f"HTTP {e.code}: {err_j.get('error', {}).get('message', err_b)}"
        except:
            return False, f"HTTP {e.code}: {err_b}"
    except Exception as e:
        return False, str(e)


def test_anthropic(api_key: str):
    if not api_key:
        return False, "Not configured in .env"
    url = "https://api.anthropic.com/v1/messages"
    payload = {
        "model": "claude-3-5-sonnet-20240620",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "ping"}]
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
            "User-Agent": "KeyVerifier/1.0"
        }
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return True, "Valid! (Successfully reached Claude API)"
    except urllib.error.HTTPError as e:
        err_b = e.read().decode("utf-8", errors="ignore")
        try:
            err_j = json.loads(err_b)
            return False, f"HTTP {e.code}: {err_j.get('error', {}).get('message', err_b)}"
        except:
            return False, f"HTTP {e.code}: {err_b}"
    except Exception as e:
        return False, str(e)


def test_huggingface(api_key: str):
    if not api_key:
        return False, "Not configured in .env"
    url = "https://huggingface.co/api/whoami-v2"
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {api_key}", "User-Agent": "KeyVerifier/1.0"}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            user = data.get("name", "Authenticated User")
            return True, f"Valid! (Authenticated as '{user}')"
    except urllib.error.HTTPError as e:
        err_b = e.read().decode("utf-8", errors="ignore")
        return False, f"HTTP {e.code}: {err_b}"
    except Exception as e:
        return False, str(e)


def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    env_file = os.path.join(root_dir, ".env")
    env = load_env(env_file)

    print("=" * 80)
    print(" 🔑 VERIFIED RAG - API KEYS VALIDATION REPORT")
    print(f" Environment Source: {env_file}")
    print("=" * 80)

    keys_to_test = [
        ("Google Gemini", "GEMINI_API_KEY", test_gemini),
        ("OpenAI", "OPENAI_API_KEY", test_openai),
        ("Anthropic Claude", "ANTHROPIC_API_KEY", test_anthropic),
        ("Hugging Face", "HUGGINGFACE_API_KEY", test_huggingface),
    ]

    for provider_name, env_key, test_fn in keys_to_test:
        raw_val = env.get(env_key, "")
        masked = mask_key(raw_val)
        print(f"\n▶ Testing {provider_name} [{env_key}] ({masked}):")
        if not raw_val:
            print("  ⚠️ Status: Not set in .env (Will use local fallback)")
            continue

        ok, msg = test_fn(raw_val)
        if ok:
            print(f"  ✅ Status: ACTIVE & VALID -> {msg}")
        else:
            print(f"  ❌ Status: INVALID / ERROR -> {msg}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
