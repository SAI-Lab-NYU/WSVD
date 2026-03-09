#!/usr/bin/env bash
set -euo pipefail

# Default usage assumes current directory is: wsvd/script/local_qat
# This script is also robust to being called from any directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_QAT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WSVD_DIR="$(cd "${LOCAL_QAT_DIR}/../.." && pwd)"
CACHE_DIR="${WSVD_DIR}/cache_file"

REPO_ID="Etropyyy/wsvd-cache"

if ! command -v hf >/dev/null 2>&1; then
  echo "Error: 'hf' command not found. Please install huggingface_hub CLI first."
  echo "Run: pip install -U \"huggingface_hub[cli]\""
  exit 1
fi

mkdir -p "${CACHE_DIR}"

echo "Downloading cache files from ${REPO_ID} ..."
echo "Target directory: ${CACHE_DIR}"
hf download "${REPO_ID}" --repo-type model --local-dir "${CACHE_DIR}"

echo "Done."
