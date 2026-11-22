from hashlib import sha256
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from runtime_cuda.downloads import DRAFTER_REPO, DRAFTER_WEIGHTS, REVISIONS, resolve_path


def test_download_selects_and_verifies_target_drafter(tmp_path):
    name = DRAFTER_WEIGHTS["5.4GB"]
    payload = b"test drafter"
    weight = tmp_path / name
    weight.write_bytes(payload)
    entry = SimpleNamespace(rfilename=name, lfs=SimpleNamespace(size=len(payload), sha256=sha256(payload).hexdigest()))
    response = SimpleNamespace(sha=REVISIONS[DRAFTER_REPO], siblings=[entry])
    with (patch("huggingface_hub.HfApi") as api,
          patch("huggingface_hub.snapshot_download", return_value=tmp_path) as download):
        api.return_value.model_info.return_value = response
        assert resolve_path("auto", drafter=True, target="5.4GB") == weight
        assert download.call_args.kwargs["revision"] == REVISIONS[DRAFTER_REPO]
        assert name in download.call_args.kwargs["allow_patterns"]
        assert DRAFTER_WEIGHTS["8.9GB"] not in download.call_args.kwargs["allow_patterns"]
        weight.write_bytes(b"X" * len(payload))
        with pytest.raises(ValueError, match="SHA-256 verification failed"):
            resolve_path("auto", drafter=True, target="5.4GB")
