from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.open_nanobrain_test_labels import (
    _encode_selected_rows,
    _filter_to_compact_train_classes,
    _verify_post_freeze_test_pairs,
)


def test_encode_selected_rows_replays_v3_source_sorted_order() -> None:
    frames = [
        pd.DataFrame(
            {
                "source_id": [4, 3, 4, 3],
                "row_index": [8, 9, 2, 1],
                "annotation_label": ["d", "b", "c", "a"],
            }
        )
    ]
    # groupby(sort=False) visits source 4 then source 3; rows are sorted inside
    # each source, matching the V3 joint-metadata materializer.
    labels, rows = _encode_selected_rows(
        frames,
        np.asarray([0, 2, 3]),
        np.asarray([4, 3, 3]),
        {"a": 0, "b": 1, "c": 2, "d": 3},
    )
    assert rows == 4
    np.testing.assert_array_equal(labels, [2, 0, 1])


def test_encode_selected_rows_fails_on_identity_mismatch() -> None:
    frame = pd.DataFrame(
        {"source_id": [2], "row_index": [7], "annotation_label": ["x"]}
    )
    with pytest.raises(ValueError, match="row identity mismatch"):
        _encode_selected_rows([frame], np.asarray([0]), np.asarray([9]), {"x": 0})


def test_post_freeze_pair_delta_rejects_unrelated_manifest_change(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"split_id":"x","splits":{"test":{"files":{},"unrelated":true}}}\n'
    )
    assert _verify_post_freeze_test_pairs(manifest, "not-a-hash") is None


def test_test_labels_exclude_classes_absent_from_compact_train() -> None:
    labels, mask, stats = _filter_to_compact_train_classes(
        np.asarray([0, 1, 2, -1]),
        np.asarray([True, True, True, True]),
        np.asarray([0, 2, -1]),
        np.asarray([True, True, False]),
    )
    np.testing.assert_array_equal(labels, [0, -1, 2, -1])
    np.testing.assert_array_equal(mask, [True, False, True, False])
    assert stats["compact_train_unseen_classes"] == {"1": 1}
