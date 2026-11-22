#!/bin/bash
# Tune a rented card over SSH and copy its table back into runtime_cuda/inkling/tuning/.
#
#   runtime_cuda/remote_tune.sh HOST PORT          # e.g. a Runpod pod's ssh.direct host and port
#
# Copies the files tune.py needs, checks the host's driver can initialise CUDA (some community hosts cannot),
# installs the PyTorch/Triton versions this repo is measured with, runs the full sweep (~45-60 min, mostly Triton
# compiles), and fetches result.json. Needs SSH key access as root (Runpod: startSsh + a registered key).
set -euo pipefail
H=$1; P=$2
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
TMP=$(mktemp -d)
O=(-o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$TMP/known_hosts" -o ConnectTimeout=20 -o ServerAliveInterval=60)
TORCH=${TORCH:-2.14.0}; INDEX=${INDEX:-https://download.pytorch.org/whl/cu130}
(cd "$ROOT" && tar czf "$TMP/bundle.tgz" runtime_cuda/__init__.py runtime_cuda/inkling/__init__.py \
    runtime_cuda/inkling/{attention,formats,gemv,rht,triton_kernels,tuning}.py runtime_cuda/tune.py)
scp "${O[@]}" -P "$P" "$TMP/bundle.tgz" root@"$H":/root/bundle.tgz
ssh "${O[@]}" -p "$P" root@"$H" TORCH="$TORCH" INDEX="$INDEX" 'bash -s' <<'REMOTE'
set -e
mkdir -p /root/lamq && cd /root/lamq && tar xzf /root/bundle.tgz
nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv,noheader
python3 -c "import ctypes, sys; r = ctypes.CDLL('libcuda.so.1').cuInit(0); print('cuInit', r); sys.exit(r)" || { echo "BAD_HOST: the driver cannot initialise CUDA"; exit 3; }
python3 -m pip install -q --no-cache-dir torch==$TORCH --index-url $INDEX 2>/dev/null \
  || python3 -m pip install -q --no-cache-dir --break-system-packages torch==$TORCH --index-url $INDEX
python3 -u runtime_cuda/tune.py --out /root/lamq/result.json
REMOTE
KEY=$(ssh "${O[@]}" -p "$P" root@"$H" "python3 -c \"import json; print(json.load(open('/root/lamq/result.json'))['gpu']['key'])\"")
scp "${O[@]}" -P "$P" root@"$H":/root/lamq/result.json "$ROOT/runtime_cuda/inkling/tuning/$KEY.json"
echo "wrote runtime_cuda/inkling/tuning/$KEY.json"
rm -rf "$TMP"
