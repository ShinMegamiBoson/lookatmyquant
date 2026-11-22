"""Resolve local packs or download and verify Hugging Face snapshots."""
from functools import lru_cache
from hashlib import sha256
from pathlib import Path

HF_PREFIX = "shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-"
MODEL_REVISIONS = {
    "3.9GB": "54b4d8c2932ae5c0fc3b8a7a8c39f2633baef67e",
    "4.7GB": "5450b7e5d0c7007d61993a7bdd5816bc346208ba",
    "5.4GB": "71a8caebb485a4ebb0f0dd994f75ae9a6b1d57e0",
    "8.9GB": "11a9b0f151e066c0f75a30c4b20ab5c48862a528",
    "11.2GB": "45d14b0f3c9bbf0989345418521385f209a39239",
    "12.8GB": "758a177fdaba30e53cce1266504b5428290aa0c9",
}
DRAFTER_REPO = HF_PREFIX + "drafter"
REVISIONS = {HF_PREFIX + size: rev for size, rev in MODEL_REVISIONS.items()}
REVISIONS[DRAFTER_REPO] = "adac74d1cf00ed68a19a77233a62bd6dcc6c2607"
DRAFTER_WEIGHTS = {
    "5.4GB": "mtp_4bit_5p4gb_v2.safetensors",
    "8.9GB": "mtp_4bit_8p9gb_v2.safetensors",
}
MODEL_FILES = ("*.json", "*.npz", "steering.safetensors", "*.jinja", "tokenizer*", "vocab.*", "merges.txt")


@lru_cache(maxsize=128)
def _sha256(path: Path, size: int, mtime: int, ctime: int) -> str:
    digest = sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify(path, expected):
    stat = path.stat()
    if stat.st_size != expected.size or _sha256(path.resolve(), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) != expected.sha256:
        raise ValueError(f"SHA-256 verification failed for {path.name}")


def resolve_path(path_or_repo, *, revision=None, drafter=False, target=None, local_dir=None):
    """Resolve a local pack, the published size name, or a Hugging Face repo ID.

    Downloaded LFS data is verified against the metadata for the resolved commit.
    For DSpark, target-aware selection applies only to the published default revision.
    """
    path = Path(path_or_repo).expanduser()
    if path.exists():
        if revision is not None:
            raise ValueError("A revision applies only to a Hugging Face repository")
        return path
    repo = str(path_or_repo)
    if repo in MODEL_REVISIONS and not drafter:
        repo = HF_PREFIX + repo
    elif repo in ("auto", "drafter") and drafter:
        repo = DRAFTER_REPO
    if repo.count("/") != 1 or repo.startswith((".", "/", "~")):
        raise FileNotFoundError(f"model path does not exist: {path_or_repo}")

    from fnmatch import fnmatch
    from huggingface_hub import HfApi, snapshot_download

    requested = revision or REVISIONS.get(repo)
    info = HfApi().model_info(repo, revision=requested, files_metadata=True)
    if not info.sha:
        raise ValueError(f"Hugging Face returned no commit for {repo}")
    weight_name = None
    if drafter and repo.lower() == DRAFTER_REPO.lower() and info.sha == REVISIONS[DRAFTER_REPO]:
        size = str(target).rstrip("/").split("/")[-1].rsplit("-", 1)[-1]
        weight_name = DRAFTER_WEIGHTS.get(size, "mtp_4bit.safetensors")
    patterns = ("config.json", "manifest.json", weight_name or "*.safetensors") if drafter else MODEL_FILES
    files = [f for f in info.siblings if any(fnmatch(f.rfilename, pattern) for pattern in patterns)]
    if weight_name and not any(f.rfilename == weight_name for f in files):
        raise ValueError(f"{repo} at {info.sha} has no {weight_name}")
    path = Path(snapshot_download(repo_id=repo, revision=info.sha, allow_patterns=patterns, local_dir=local_dir))
    for file in files:
        if file.lfs is not None:
            _verify(path / file.rfilename, file.lfs)
    return path / weight_name if weight_name else path
