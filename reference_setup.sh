#!/usr/bin/env bash
# Reference / Alternative Generator Script for Verified RAG
# Kept as a side reference script.

set -euo pipefail

# ----------------------------------------------------------------------
# Helper: write a file, creating parent dirs automatically
# ----------------------------------------------------------------------
write_file() {
  local path=$1
  local content=$2
  mkdir -p "$(dirname "$path")"
  printf "%s" "$content" > "$path"
}

BASE=verified-rag-reference

# .gitignore
write_file "$BASE/.gitignore" "$(cat <<'EOF'
node_modules/
.next/
dist/
.env
*.log
__pycache__/
*.pyc
*.pyo
*.egg-info/
.vscode/
docker-compose.override.yml
EOF
)"

# .env.example
write_file "$BASE/.env.example" "$(cat <<'EOF'
SUPABASE_URL=http://localhost:5432
SUPABASE_ANON_KEY=YOUR_ANON_KEY
SUPABASE_SERVICE_ROLE_KEY=YOUR_SERVICE_ROLE_KEY
POSTGRES_PASSWORD=postgres

ANTHROPIC_API_KEY=sk-ant-...

EMBED_SERVICE_URL=http://embed-service:8000
RERANK_SERVICE_URL=http://rerank-service:8000
VERIF_SERVICE_URL=http://verification-service:8000
EOF
)"

echo "Reference script saved."
