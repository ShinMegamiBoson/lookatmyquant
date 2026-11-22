"""Lists upstream changes to the MLX runtime since the commit recorded in runtime_cuda/SYNC.md.

    python runtime_cuda/upstream_sync.py            # against the origin/main you have
    python runtime_cuda/upstream_sync.py --fetch    # git fetch origin first
    python runtime_cuda/upstream_sync.py --diff     # also print each behaviour file's diff

Each changed file is sorted into "behaviour" (port it: formats, model semantics, decoding) or "Metal speed" (read
it; usually nothing to do here), with the files in runtime_cuda/ that correspond (from SYNC.md's file map).
"""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYNC = ROOT / "runtime_cuda" / "SYNC.md"

# upstream path -> (kind, our counterparts). kernels.py mixes reference math (behaviour) and Metal source.
MAP = {
    "runtime_mlx/inkling/model.py": ("behaviour", "model.py, schema.py, formats.py, reference.py"),
    "runtime_mlx/inkling/kernels.py": ("behaviour", "formats.py, gemv.py (reference math, new formats)"),
    "runtime_mlx/inkling/fast_kernels.py": ("Metal speed", "gemv.py, triton_kernels.py, attention.py, conv.py, fused.py, rht.py"),
    "runtime_mlx/inkling/generate.py": ("behaviour", "generate.py"),
    "runtime_mlx/inkling/drafter.py": ("behaviour", "(not ported: DSpark drafter)"),
    "runtime_mlx/test_inkling.py": ("behaviour", "test_upstream_fixture.py and the tests the new cases imply"),
    "runtime_mlx/runtime.py": ("behaviour", "(not ported: lamq Runtime integration)"),
    "runtime_mlx/cli.py": ("behaviour", "(not ported: lamq CLI)"),
}


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout


def synced_commit() -> str:
    m = re.search(r"\*\*Synced through upstream commit:\*\*\s*`([0-9a-f]{7,40})`", SYNC.read_text())
    if not m:
        raise SystemExit(f"no 'Synced through upstream commit' line in {SYNC}")
    return m.group(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--diff", action="store_true")
    ap.add_argument("--upstream", default="origin/main")
    args = ap.parse_args()
    if args.fetch:
        git("fetch", "-q", "origin")
    base = synced_commit()
    commits = git("log", "--reverse", "--format=%h %ad %s", "--date=short", f"{base}..{args.upstream}", "--", "runtime_mlx").splitlines()
    print(f"upstream {args.upstream} vs synced {base}: {len(commits)} commit(s) touching runtime_mlx/")
    for line in commits:
        sha = line.split()[0]
        print(f"\n{line}")
        files = git("show", "--format=", "--numstat", sha, "--", "runtime_mlx").splitlines()
        for f in files:
            add, rem, path = f.split("\t")
            kind, ours = MAP.get(path, ("behaviour" if "inkling" in path else "other model", "(check: not in the file map)"))
            print(f"  [{kind:11s}] {path} (+{add} -{rem})  ->  {ours}")
            if args.diff and kind == "behaviour":
                print(git("show", "--format=", sha, "--", path))
    if not commits:
        print("nothing to port")


if __name__ == "__main__":
    main()
