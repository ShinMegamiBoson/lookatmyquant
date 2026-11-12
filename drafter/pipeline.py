"""python -m drafter.pipeline --root ROOT --stage STAGE [--smoke]

ROOT is a folder on a persistent volume. The stages, in order: assets, text, baseline, init, features, train, score
(`--stage all` runs them all). A finished stage leaves ROOT/done/<stage>.json and is skipped from then on; `features`
writes to local disk, so its marker is there and a fresh container builds the features again. An unfinished stage carries
on from the shards, files and checkpoints it already has. Nothing under ROOT is deleted. ROOT/status.json says where
the run is. Exit code 0 means every requested stage is done."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

from . import common, settings

STAGES = {
    "assets": ("assets", "run"),
    "text": ("text", "run"),
    "baseline": ("score", "run_baseline"),
    "init": ("init", "run"),
    "features": ("features", "run"),
    "train": ("train", "run"),
    "score": ("score", "run_score"),
}


def _marker(root: Path, s, stage: str) -> Path:
    return Path(s.features_dir) / "done.json" if stage == "features" else root / "done" / f"{stage}.json"


def _done(root: Path, s, stage: str) -> bool:
    marker = _marker(root, s, stage)
    return marker.exists() and (stage != "features" or common.read_json(marker)["root"] == str(root))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True)
    parser.add_argument("--stage", required=True, choices=[*STAGES, "all"])
    parser.add_argument("--smoke", action="store_true", help="the same run at toy size")
    args = parser.parse_args(argv)

    s = settings.Settings().smoke() if args.smoke else settings.Settings()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    status = common.Status(root)
    for stage in STAGES if args.stage == "all" else [args.stage]:
        status.begin(stage)
        if _done(root, s, stage):
            status.end("done", "already done, skipped")
            continue
        module, function = STAGES[stage]
        try:
            result = getattr(importlib.import_module(f"{__package__}.{module}"), function)(root, s, status)
        except BaseException as error:
            status.end("failed", f"failed: {error!r}")
            raise
        common.write_json(_marker(root, s, stage), {"stage": stage, "root": str(root), "smoke": args.smoke,
                                                    "seconds": status.seconds(), **(result or {})})
        status.end("done", f"done in {status.seconds():.0f} s")
        settings.COMMIT()
    return 0


if __name__ == "__main__":
    sys.exit(main())
