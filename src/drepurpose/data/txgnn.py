__all__ = (
    "DISEASE_AREAS",
    "THERAPEUTIC_RELATIONS",
    "DiseaseArea",
    "TxGNNSplit",
    "build_disease_area_split",
)

from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import pandas as pd
import torch
from torch_geometric.utils import k_hop_subgraph

DiseaseArea = Literal[
    "adrenal_gland",
    "anemia",
    "cardiovascular",
    "cell_proliferation",
    "mental_health",
]

DISEASE_AREAS: dict[DiseaseArea, str] = {
    "adrenal_gland": "9553",
    "anemia": "2355",
    "cardiovascular": "1287",
    "cell_proliferation": "14566",
    "mental_health": "150",
}

THERAPEUTIC_RELATIONS = (
    "contraindication",
    "indication",
    "off-label use",
)

_EDGE_COLUMNS = (
    "x_index",
    "y_index",
    "relation",
)


class TxGNNSplit(TypedDict):
    nodes: pd.DataFrame
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    ontology_disease_count: int
    sampled_test_pair_count: int
    directed_edge_count: int
    final_disease_count: int


def _obo_children(path: Path) -> dict[str, set[str]]:
    children: dict[str, set[str]] = {}
    term: str | None = None

    with path.open(encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()

            if line == "[Term]":
                term = None
            elif line.startswith("id: DOID:"):
                term = line.removeprefix("id: DOID:")
            elif term is not None and line.startswith("is_a: DOID:"):
                parent = line.removeprefix("is_a: DOID:").split(maxsplit=1)[0]
                children.setdefault(parent, set()).add(term)

    return children


def _descendants(children: dict[str, set[str]], root: str) -> set[str]:
    descendants = {parent: values.copy() for parent, values in children.items()}

    for _ in range(20):
        for parent, values in descendants.items():
            grandchildren: set[str] = set()

            for child in values:
                grandchildren.update(descendants.get(child, set()))

            descendants[parent] = values | grandchildren

    return {root, *descendants.get(root, set())}


def _indexed_kg(edges: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    if not nodes.node_index.is_unique:
        raise RuntimeError("PrimeKG node indices are not unique")

    node_indices = pd.Index(nodes.node_index)
    missing_x = ~edges.x_index.isin(node_indices)
    missing_y = ~edges.y_index.isin(node_indices)

    if missing_x.any() or missing_y.any():
        raise RuntimeError(
            "PrimeKG edges reference unknown nodes: "
            f"{int(missing_x.sum()):,} sources, {int(missing_y.sum()):,} targets"
        )

    node_metadata = nodes.set_index("node_index")[["node_id", "node_type"]]
    x_metadata = node_metadata.rename(columns={"node_id": "x_id", "node_type": "x_type"})
    y_metadata = node_metadata.rename(columns={"node_id": "y_id", "node_type": "y_type"})

    return (
        edges.join(x_metadata, on="x_index", validate="many_to_one")
        .join(y_metadata, on="y_index", validate="many_to_one")
        .loc[
            :,
            [
                "x_index",
                "x_type",
                "x_id",
                "relation",
                "y_index",
                "y_type",
                "y_id",
            ],
        ]
    )


def _disease_nodes(raw_root: Path, root: str, nodes: pd.DataFrame) -> np.ndarray:
    txgnn = raw_root / "txgnn-1000aac"
    doids = _descendants(_obo_children(txgnn / "HumanDO.obo"), root)

    xrefs = pd.read_csv(txgnn / "mondo_references.csv", low_memory=False)
    mondo = (
        xrefs.loc[
            (xrefs.ontology == "DOID") & xrefs.ontology_id.astype(str).isin(doids),
            "mondo_id",
        ]
        .drop_duplicates()
        .astype(str)
        .to_numpy()
    )

    direct = nodes.loc[
        nodes.node_id.isin(mondo) & (nodes.node_source == "MONDO"), "node_index"
    ].to_numpy()

    groups = pd.read_csv(txgnn / "kg_grouped_diseases_bert_map.csv", low_memory=False)
    group_ids = (
        groups.loc[
            groups.node_id.isin(mondo) & (groups.node_source == "MONDO"),
            "group_id_bert",
        ]
        .drop_duplicates()
        .astype(str)
        .to_numpy()
    )

    grouped = nodes.loc[
        nodes.node_id.isin(group_ids) & (nodes.node_source == "MONDO_grouped"),
        "node_index",
    ].to_numpy()

    disease_nodes = np.unique(np.concatenate((direct, grouped))).astype(np.int64)

    if not len(disease_nodes):
        raise RuntimeError(f"No PrimeKG nodes resolved for DOID {root}")

    return disease_nodes


def _test_pairs(edges: pd.DataFrame, disease_nodes: np.ndarray, seed: int) -> pd.DataFrame:
    therapeutic = (
        edges.loc[
            (edges.x_index.isin(disease_nodes) | edges.y_index.isin(disease_nodes))
            & edges.relation.isin(THERAPEUTIC_RELATIONS),
            ["x_index", "y_index"],
        ]
        .to_numpy()
        .T
    )

    edge_index = torch.as_tensor(edges[["x_index", "y_index"]].to_numpy().T, dtype=torch.long)
    _, local_edges, _, _ = k_hop_subgraph(disease_nodes.tolist(), 2, edge_index)

    random_count = round(edge_index.shape[1] * 0.05) - therapeutic.shape[1]
    if not 0 <= random_count <= local_edges.shape[1]:
        raise RuntimeError(
            f"Cannot sample {random_count:,} edges from a 2-hop region with "
            f"{local_edges.shape[1]:,} edges"
        )

    rng = np.random.RandomState(seed)
    sampled_indices = rng.choice(local_edges.shape[1], random_count, replace=False)
    sampled = local_edges[:, sampled_indices].numpy()
    pairs = np.unique(np.concatenate((therapeutic, sampled), axis=1), axis=1)

    return pd.DataFrame(pairs.T, columns=["x_index", "y_index"])


def _mark_test_region(kg: pd.DataFrame, test_pairs: pd.DataFrame) -> pd.DataFrame:
    matched = test_pairs.merge(
        kg[["x_index", "y_index"]].drop_duplicates(),
        on=["x_index", "y_index"],
        how="left",
        indicator=True,
    )

    missing = int((matched._merge == "left_only").sum())
    if missing:
        raise RuntimeError(f"{missing:,} sampled edge pairs are absent from PrimeKG")

    train = kg.copy()
    train["split"] = "train"

    test = kg.merge(test_pairs, on=["x_index", "y_index"], how="inner")
    test["split"] = "test"

    return (
        pd.concat((train, test))
        .drop_duplicates(subset=["x_index", "y_index"], keep="last")
        .reset_index(drop=True)
    )


def _convert_id(value: object) -> str:
    text = str(value)

    try:
        if "_" not in text:
            return str(float(text))
    except ValueError:
        pass

    return text


def _directed_kg(frame: pd.DataFrame) -> pd.DataFrame:
    keep: list[int] = []

    for relation in np.unique(frame.relation.to_numpy()):
        relation_frame = frame.loc[frame.relation == relation].copy()
        parts = relation.split("_")

        if len(parts) > 1 and parts[0] == parts[1]:
            relation_frame["_pair"] = relation_frame.apply(
                lambda row: "_".join(sorted((str(row.x_id), str(row.y_id)))), axis=1
            )
            keep.extend(relation_frame.drop_duplicates("_pair").index.tolist())
        else:
            forward_type = relation_frame.x_type.iloc[0]
            keep.extend(relation_frame.loc[relation_frame.x_type == forward_type].index.tolist())

    directed = frame.loc[frame.index.isin(keep)].copy()
    directed["x_id"] = directed.x_id.map(_convert_id)
    directed["y_id"] = directed.y_id.map(_convert_id)
    directed["x_idx"] = np.nan
    directed["y_idx"] = np.nan

    node_types = np.unique(np.concatenate((directed.x_type.unique(), directed.y_type.unique())))

    for node_type in node_types:
        ids = np.unique(
            np.concatenate(
                (
                    directed.loc[directed.x_type == node_type, "x_id"].to_numpy(),
                    directed.loc[directed.y_type == node_type, "y_id"].to_numpy(),
                )
            )
        )

        mapping = dict(zip(ids, range(len(ids)), strict=True))
        x_mask = directed.x_type == node_type
        y_mask = directed.y_type == node_type

        directed.loc[x_mask, "x_idx"] = directed.loc[x_mask, "x_id"].map(mapping)
        directed.loc[y_mask, "y_idx"] = directed.loc[y_mask, "y_id"].map(mapping)

    directed["x_idx"] = directed.x_idx.astype(np.int64)
    directed["y_idx"] = directed.y_idx.astype(np.int64)

    return directed.reset_index(drop=True)


def _train_valid_test(
    frame: pd.DataFrame, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_val = frame.loc[frame.split == "train"].reset_index(drop=True)
    test = frame.loc[frame.split == "test"].reset_index(drop=True)
    train_parts: list[pd.DataFrame] = []
    valid_parts: list[pd.DataFrame] = []

    for relation in train_val.relation.unique():
        group = train_val.loc[train_val.relation == relation]

        unused_test = group.sample(frac=0.0, replace=False, random_state=seed)
        remaining = group.loc[~group.index.isin(unused_test.index)]

        valid = remaining.sample(frac=0.125, replace=False, random_state=1)
        train_parts.append(remaining.loc[~remaining.index.isin(valid.index)])
        valid_parts.append(valid)

    return (
        pd.concat(train_parts).reset_index(drop=True),
        pd.concat(valid_parts).reset_index(drop=True),
        test,
    )


def _reverse_test(test: pd.DataFrame, directed: pd.DataFrame) -> pd.DataFrame:
    relations = directed[["x_type", "relation", "y_type"]].drop_duplicates()
    parts = [test]

    for x_type, relation, y_type in relations.itertuples(index=False, name=None):
        reverse = test.loc[test.relation == relation].rename(
            columns={
                "x_type": "y_type",
                "x_id": "y_id",
                "x_idx": "y_idx",
                "y_type": "x_type",
                "y_id": "x_id",
                "y_idx": "x_idx",
            }
        )

        if x_type != y_type:
            reverse["relation"] = f"rev_{relation}"

        parts.append(reverse)

    return pd.concat(parts).reset_index(drop=True)


def _filter_test_diseases(
    raw_root: Path,
    area: DiseaseArea,
    directed: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    disease_list = pd.read_csv(
        raw_root / "txgnn-1000aac" / "disease_files" / f"{area}.csv",
        low_memory=False,
    )

    id_to_idx = dict(
        directed.loc[directed.x_type == "disease", ["x_id", "x_idx"]].drop_duplicates().values
    )
    id_to_idx.update(
        dict(directed.loc[directed.y_type == "disease", ["y_id", "y_idx"]].drop_duplicates().values)
    )

    merged: dict[str, int] = {}

    for node_id, node_idx in id_to_idx.items():
        try:
            members = node_id.split("_") if "_" in node_id else (node_id,)

            for member in members:
                merged[str(float(member))] = int(node_idx)
        except ValueError:
            continue

    id_to_idx.update(merged)

    disease_list["node_idx"] = disease_list.node_id.map(
        lambda value: id_to_idx.get(_convert_id(value), -1)
    )
    target_indices = disease_list.loc[disease_list.node_idx >= 0, "node_idx"].unique()

    reverse_relations = tuple(f"rev_{relation}" for relation in THERAPEUTIC_RELATIONS)
    reverse_therapeutic = test.relation.isin(reverse_relations)
    wrong_disease = reverse_therapeutic & ~test.x_idx.isin(target_indices)

    return test.loc[~wrong_disease].reset_index(drop=True), len(target_indices)


def build_disease_area_split(raw_root: Path, area: DiseaseArea, seed: int) -> TxGNNSplit:
    primekg = raw_root / "primekg"

    nodes = pd.read_csv(primekg / "nodes.csv", low_memory=False)
    edges = pd.read_csv(primekg / "edges.csv", usecols=_EDGE_COLUMNS, low_memory=False)
    kg = _indexed_kg(edges, nodes)

    disease_nodes = _disease_nodes(raw_root, DISEASE_AREAS[area], nodes)
    test_pairs = _test_pairs(edges, disease_nodes, seed)
    directed = _directed_kg(_mark_test_region(kg, test_pairs))

    train, valid, test = _train_valid_test(directed, seed)
    test = _reverse_test(test, directed)
    test, final_disease_count = _filter_test_diseases(raw_root, area, directed, test)

    return {
        "nodes": nodes,
        "train": train,
        "valid": valid,
        "test": test,
        "ontology_disease_count": len(disease_nodes),
        "sampled_test_pair_count": len(test_pairs),
        "directed_edge_count": len(directed),
        "final_disease_count": final_disease_count,
    }
