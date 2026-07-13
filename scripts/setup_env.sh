#!/usr/bin/env bash
# Build the `octotools` conda env (py3.10, external-vLLM setup: no local vllm).
set -eo pipefail

ENV_NAME="${1:-octotools}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" python=3.10 pip
fi
conda activate "${ENV_NAME}"
# guard against the create silently failing and pip hitting another env
if [[ "$CONDA_PREFIX" != *"/envs/${ENV_NAME}" ]]; then
  echo "ERROR: expected env ${ENV_NAME}, got CONDA_PREFIX=$CONDA_PREFIX" >&2
  exit 1
fi

# vllm==0.8.5 is excluded: we talk to an externally hosted vLLM server via
# VLLM_BASE_URL, so the local vllm package (and its torch pin) is unnecessary.
grep -v '^vllm==' "${REPO_ROOT}/requirements.txt" > /tmp/octotools-reqs-novllm.txt
pip install -r /tmp/octotools-reqs-novllm.txt
pip install -e "${REPO_ROOT}" --no-deps

echo "OCTOTOOLS_ENV_READY"
