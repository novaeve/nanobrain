import json

import numpy as np

from nanobrain.paths import SPLIT_ID, describe_v3_shards, locate_v3_shards


def _fake_corpus(root):
    shard_root = root / "assets" / "v3_numpy_shards"
    shard_root.mkdir(parents=True)
    for index, split in enumerate(("train", "validation", "test"), start=1):
        (shard_root / f"{split}-manifest.json").write_text(
            json.dumps({
                "split_id": SPLIT_ID,
                "split": split,
                "rows": index,
                "shards": 1,
                "complete": True,
            })
        )
        stem = shard_root / f"{split}-shard-00000"
        np.save(f"{stem}-row_ptr.npy", np.array([0, 1], dtype=np.int64))
        np.save(f"{stem}-gene_index.npy", np.array([index], dtype=np.int32))
        np.save(f"{stem}-values.npy", np.array([float(index)], dtype=np.float32))
    return shard_root


def test_locator_prefers_and_inventories_project_corpus(tmp_path):
    expected = _fake_corpus(tmp_path)
    location = locate_v3_shards(project_root=tmp_path)
    assert location.root == expected
    assert location.rows == {"train": 1, "validation": 2, "test": 3}
    assert location.total_rows == 6

    inventory = describe_v3_shards(project_root=tmp_path)
    assert inventory["complete"] is True
    assert inventory["h5ad_in_training_loop"] is False
    assert inventory["project_alias"] == str(expected)
    assert inventory["split_file_count"] == {
        "train": 3,
        "validation": 3,
        "test": 3,
    }
    assert inventory["bytes_on_disk"] == sum(inventory["bytes"].values()) + sum(
        (expected / f"{split}-manifest.json").stat().st_size
        for split in ("train", "validation", "test")
    )
