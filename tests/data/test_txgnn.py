import pandas as pd
import pytest

from drepurpose.data.txgnn import _convert_id, _descendants, _indexed_kg, _train_valid_test


def test_descendants_include_root_and_transitive_children() -> None:
    children = {"1": {"2", "3"}, "2": {"4"}, "4": {"5"}}

    assert _descendants(children, "1") == {"1", "2", "3", "4", "5"}


def test_indexed_kg_joins_edge_indices_to_nodes() -> None:
    nodes = pd.DataFrame(
        {
            "node_index": [0, 1, 2],
            "node_id": ["drug-a", "disease-a", "gene-a"],
            "node_type": ["drug", "disease", "gene/protein"],
        }
    )
    edges = pd.DataFrame(
        {
            "x_index": [0, 2],
            "y_index": [1, 1],
            "relation": ["indication", "disease_protein"],
        }
    )

    kg = _indexed_kg(edges, nodes)

    assert kg.to_dict("records") == [
        {
            "x_index": 0,
            "x_type": "drug",
            "x_id": "drug-a",
            "relation": "indication",
            "y_index": 1,
            "y_type": "disease",
            "y_id": "disease-a",
        },
        {
            "x_index": 2,
            "x_type": "gene/protein",
            "x_id": "gene-a",
            "relation": "disease_protein",
            "y_index": 1,
            "y_type": "disease",
            "y_id": "disease-a",
        },
    ]


def test_indexed_kg_rejects_unknown_node_indices() -> None:
    nodes = pd.DataFrame(
        {
            "node_index": [0],
            "node_id": ["drug-a"],
            "node_type": ["drug"],
        }
    )
    edges = pd.DataFrame(
        {
            "x_index": [0],
            "y_index": [999],
            "relation": ["indication"],
        }
    )

    with pytest.raises(RuntimeError, match="unknown nodes"):
        _indexed_kg(edges, nodes)


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
