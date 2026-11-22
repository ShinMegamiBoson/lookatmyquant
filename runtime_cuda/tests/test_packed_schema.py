import json

import numpy as np
import pytest

from runtime_cuda.inkling.schema import UnsupportedPackedFile, validate_packed_dir
from runtime_cuda.inkling_cuda.model import Model

from .fixture import build_mini_packed_dir, convert_row_groups


def test_row_group_pack_validates_and_preserves_legacy_weights(tmp_path):
    build_mini_packed_dir(tmp_path)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text())
    cfg = config["text_config"]
    cfg["sliding_window_size"] = cfg["rel_extent"]
    cfg["layer_types"] = [
        "hybrid_sliding" if layer in cfg["local_layer_ids"] else "hybrid"
        for layer in range(cfg["num_hidden_layers"])
    ]
    config_path.write_text(json.dumps(config))
    original = Model(tmp_path, None).session(4)
    before = original.step(19)

    convert_row_groups(tmp_path)
    assert "lm8" in validate_packed_dir(tmp_path)
    actual = Model(tmp_path, None).session(4).step(19)
    np.testing.assert_allclose(actual, before, rtol=0, atol=1e-6)

    layer = tmp_path / "layer_00_packed.npz"
    with np.load(layer) as packed:
        arrays = dict(packed)
    arrays["gu_rg_map"][0, 1, 1] = 32767
    np.savez(layer, **arrays)
    with pytest.raises(UnsupportedPackedFile, match="index outside the slab"):
        validate_packed_dir(tmp_path)
