import pandas as pd

from drepurpose.data.txgnn import _convert_id, _descendants, _train_valid_test


def test_descendants_include_root_and_transitive_children() -> None:
    children = {"1": {"2", "3"}, "2": {"4"}, "4": {"5"}}

    assert _descendants(children, "1") == {"1", "2", "3", "4", "5"}


def test_convert_id_matches_txgnn_normalization() -> None:
    assert _convert_id(123) == "123.0"
    assert _convert_id("123") == "123.0"
    assert _convert_id("123_456") == "123_456"
    assert _convert_id("CHEBI:15365") == "CHEBI:15365"


def test_validation_partition_uses_txgnn_fixed_seed() -> None:
    frame = pd.DataFrame(
        {
            "relation": ["relation"] * 32,
            "split": ["train"] * 32,
            "x_index": range(32),
        }
    )

    _, valid_a, _ = _train_valid_test(frame, seed=42)
    _, valid_b, _ = _train_valid_test(frame, seed=1234)

    assert len(valid_a) == 4
    assert valid_a.x_index.tolist() == valid_b.x_index.tolist()
