import pandas as pd

from drepurpose.data.prepare import _partition, _test_relation


def test_test_relation_restores_forward_orientation() -> None:
    test = pd.DataFrame(
        {
            "x_index": [10],
            "x_type": ["disease"],
            "relation": ["rev_indication"],
            "y_index": [20],
            "y_type": ["drug"],
        }
    )

    result = _test_relation(test, "indication")

    assert result.to_dict("records") == [
        {
            "x_index": 10,
            "x_type": "drug",
            "relation": "indication",
            "y_index": 20,
            "y_type": "disease",
        }
    ]


def test_partition_separates_background_and_therapeutic_edges() -> None:
    columns = [
        "x_index",
        "x_type",
        "relation",
        "y_index",
        "y_type",
    ]

    train = pd.DataFrame(
        [
            (1, "gene/protein", "protein_protein", 2, "gene/protein"),
            (10, "drug", "indication", 20, "disease"),
        ],
        columns=columns,
    )
    valid = pd.DataFrame(
        [
            (3, "disease", "disease_disease", 4, "disease"),
            (11, "drug", "contraindication", 21, "disease"),
        ],
        columns=columns,
    )
    test = pd.DataFrame(
        [
            (12, "disease", "rev_indication", 22, "drug"),
        ],
        columns=columns,
    )

    split = {
        "nodes": pd.DataFrame(),
        "train": train,
        "valid": valid,
        "test": test,
        "ontology_disease_count": 1,
        "sampled_test_pair_count": 1,
        "directed_edge_count": 5,
        "final_disease_count": 1,
    }

    parts = _partition(split)

    assert parts["background"].relation.tolist() == [
        "protein_protein",
        "disease_disease",
    ]
    assert parts["train"].relation.tolist() == ["indication"]
    assert parts["valid"].relation.tolist() == ["contraindication"]
    assert parts["test_indication"].relation.tolist() == ["indication"]
