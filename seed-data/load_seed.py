#!/usr/bin/env python3
"""
Seed Data Ingestion CLI (Pure Python Standard Library - Zero Dependencies)
Reads sample research documents from the docs/ directory and POSTs them to the /api/ingest endpoint.
"""

import os
import sys
import glob
import time
import json
import argparse
import urllib.request
import urllib.error

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

DEFAULT_ENDPOINT = os.getenv("INGEST_API_URL", "http://localhost:3000/api/ingest")
DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")


def clean_title_from_filename(filepath: str) -> str:
    base = os.path.basename(filepath)
    name = os.path.splitext(base)[0]
    return name.replace("_", " ").title()


def ingest_file(filepath: str, endpoint: str) -> dict:
    title = clean_title_from_filename(filepath)
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        return {
            "status": "error",
            "title": title,
            "filename": os.path.basename(filepath),
            "error": f"Failed to read file: {e}"
        }

    payload = {
        "title": title,
        "content": content,
        "source": os.path.basename(filepath),
        "metadata": {
            "category": "seed_dataset",
            "file_size": len(content),
            "filename": os.path.basename(filepath)
        }
    }

    start = time.time()
    req_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=req_data,
        headers={"Content-Type": "application/json", "User-Agent": "VerifiedRAG-SeedLoader/1.0"}
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            resp_body = response.read().decode("utf-8")
            data = json.loads(resp_body)
            latency = round((time.time() - start) * 1000, 2)
            return {
                "status": "success",
                "title": title,
                "filename": os.path.basename(filepath),
                "chunks": data.get("total_chunks", 0),
                "doc_id": data.get("document_id", "N/A"),
                "latency_ms": latency
            }
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode("utf-8", errors="ignore") if hasattr(e, "read") else str(e)
        return {
            "status": "error",
            "title": title,
            "filename": os.path.basename(filepath),
            "error": f"HTTP {e.code}: {err_msg}"
        }
    except urllib.error.URLError as e:
        return {
            "status": "error",
            "title": title,
            "filename": os.path.basename(filepath),
            "error": f"Connection Error (Is Next.js running on {endpoint}?): {e.reason}"
        }
    except Exception as e:
        return {
            "status": "error",
            "title": title,
            "filename": os.path.basename(filepath),
            "error": str(e)
        }


def main():
    parser = argparse.ArgumentParser(description="Verified RAG Seed Data Ingestion Tool")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="Ingestion API endpoint URL")
    parser.add_argument("--dir", default=DOCS_DIR, help="Directory containing sample documents")
    args = parser.parse_args()

    print("=" * 70)
    print(" 🚀 Verified RAG - Seed Data Ingestion Pipeline")
    print(f" Target Endpoint: {args.endpoint}")
    print(f" Document Source: {args.dir}")
    print("=" * 70)

    files = sorted(glob.glob(os.path.join(args.dir, "*.txt")) + glob.glob(os.path.join(args.dir, "*.pdf")))

    if not files:
        print(f"❌ No documents found in {args.dir}")
        sys.exit(1)

    print(f"Found {len(files)} documents to ingest.\n")
    results = []

    for idx, filepath in enumerate(files, start=1):
        print(f"[{idx}/{len(files)}] Ingesting '{os.path.basename(filepath)}'...", end=" ", flush=True)
        res = ingest_file(filepath, args.endpoint)
        if res["status"] == "success":
            print(f"✅ OK ({res['chunks']} chunks, {res['latency_ms']} ms)")
        else:
            print(f"❌ Failed: {res.get('error')}")
        results.append(res)
        time.sleep(0.05)

    print("\n" + "=" * 70)
    print(" 📋 Ingestion Summary Report")
    print("=" * 70)
    successful = sum(1 for r in results if r["status"] == "success")
    total_chunks = sum(r.get("chunks", 0) for r in results if r["status"] == "success")
    print(f" Total Ingested: {successful}/{len(files)} documents")
    print(f" Total Chunks Created & Vectorized: {total_chunks}")
    print("=" * 70)


if __name__ == "__main__":
    main()
