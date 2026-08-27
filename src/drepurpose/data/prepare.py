__all__ = ("prepare_benchmark",)

import json
import shutil
from pathlib import Path

import pandas as pd

from .fetch import audit_sources, sha256_file
from .sources import TXGNN_COMMIT
from .txgnn import (
    DISEASE_AREAS,
    THERAPEUTIC_RELATIONS,
    DiseaseArea,
    TxGNNSplit,
    build_disease_area_split,
)

_EDGE_COLUMNS = ("x_index", "x_type", "relation", "y_index", "y_type")
_EXPECTED = {
    "adrenal_gland": {"diseases": 6, "contraindication": 303, "indication": 33},
    "anemia": {"diseases": 19, "contraindication": 752, "indication": 88},
    "cardiovascular": {"diseases": 111, "contraindication": 4215, "indication": 453},
    "cell_proliferation": {"diseases": 201, "contraindication": 1047, "indication": 999},
    "mental_health": {"diseases": 60, "contraindication": 1567, "indication": 355},
}


def _test_relation(test: pd.DataFrame, relation: str) -> pd.DataFrame:
    frame = test.loc[test.relation == f"rev_{relation}"].copy()
    frame["relation"] = relation
    frame[["x_type", "y_type"]] = frame[["y_type", "x_type"]]
    return frame[list(_EDGE_COLUMNS)].reset_index(drop=True)


def _partition(split: TxGNNSplit) -> dict[str, pd.DataFrame]:
    train = split["train"]
    valid = split["valid"]

    train_mask = train.relation.isin(THERAPEUTIC_RELATIONS)
    valid_mask = valid.relation.isin(THERAPEUTIC_RELATIONS)

    background = pd.concat((train.loc[~train_mask], valid.loc[~valid_mask]))

    return {
        "background": background[list(_EDGE_COLUMNS)],  # pyright: ignore[reportReturnType]
        "train": train.loc[train_mask, list(_EDGE_COLUMNS)],
        "valid": valid.loc[valid_mask, list(_EDGE_COLUMNS)],
        "test_indication": _test_relation(split["test"], "indication"),
        "test_contraindication": _test_relation(split["test"], "contraindication"),
        "test_off_label": _test_relation(split["test"], "off-label use"),
    }


def _edge_keys(frame: pd.DataFrame) -> set[tuple[int, str, int]]:
    return {
        (int(x), str(relation), int(y))
        for x, relation, y in frame[["x_index", "relation", "y_index"]].itertuples(
            index=False, name=None
        )
    }


def _validate(area: DiseaseArea, disease_count: int, parts: dict[str, pd.DataFrame]) -> None:
    actual = {
        "diseases": disease_count,
        "contraindication": len(parts["test_contraindication"]),
        "indication": len(parts["test_indication"]),
    }
    if actual != _EXPECTED[area]:
        raise RuntimeError(f"Benchmark count mismatch: {actual} != {_EXPECTED[area]}")

    if parts["background"].relation.isin(THERAPEUTIC_RELATIONS).any():
        raise RuntimeError("Therapeutic edges in background")

    test = pd.concat(
        (parts["test_indication"], parts["test_contraindication"], parts["test_off_label"])
    )

    for name in ("train", "valid", "test"):
        frame = test if name == "test" else parts[name]

        if (~frame.relation.isin(THERAPEUTIC_RELATIONS)).any():
            raise RuntimeError(f"Non-therapeutic edges in {name}")

        if ((frame.x_type != "drug") | (frame.y_type != "disease")).any():
            raise RuntimeError(f"Invalid therapeutic edges in {name}")

    held_out = test.y_index.unique()
    for name in ("train", "valid"):
        if parts[name].y_index.isin(held_out).any():
            raise RuntimeError(f"Held-out diseases in {name}")

    train = _edge_keys(parts["train"])
    valid = _edge_keys(parts["valid"])
    test = _edge_keys(test)

    if train & valid or train & test or valid & test:
        raise RuntimeError("Therapeutic partitions overlap")


def _write(
    path: Path,
    nodes: pd.DataFrame,
    parts: dict[str, pd.DataFrame],
    manifest: dict[str, object],
) -> None:
    path.mkdir(parents=True)
    files: dict[str, object] = {}

    for name, frame in {"nodes": nodes, **parts}.items():
        output = (
            frame.sort_values("node_index")
            if name == "nodes"
            else frame.sort_values(["relation", "x_index", "y_index"])
        )

        file = path / f"{name}.parquet"
        output.to_parquet(file, index=False, compression="zstd")

        files[name] = {
            "path": file.name,
            "rows": len(output),
            "sha256": sha256_file(file),
        }

    manifest["files"] = files
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def prepare_benchmark(
    raw_root: Path,
    output_root: Path,
    area: DiseaseArea,
    *,
    seed: int = 42,
    force: bool = False,
) -> Path:
    path = output_root / area / f"seed-{seed}"
    temporary = path.with_name(f"{path.name}.part")

    if path.exists() and not force:
        raise FileExistsError(path)
    if path.exists():
        raise FileExistsError(path)

    shutil.rmtree(temporary, ignore_errors=True)

    shutil.rmtree(temporary, ignore_errors=True)

    source_manifest = raw_root / "manifest.json"
    if not source_manifest.exists():
        raise FileNotFoundError(source_manifest)

    audit_sources(raw_root)

    split = build_disease_area_split(raw_root, area, seed)
    parts = _partition(split)
    disease_count = split["final_disease_count"]

    _validate(area, disease_count, parts)

    nodes = split["nodes"]
    manifest: dict[str, object] = {
        "format_version": 1,
        "dataset": "PrimeKG",
        "benchmark": "TxGNN/BioPathNet zero-shot disease-area",
        "disease_area": area,
        "disease_ontology_root": DISEASE_AREAS[area],
        "seed": seed,
        "txgnn_commit": TXGNN_COMMIT,
        "source_manifest_sha256": sha256_file(source_manifest),
        "counts": {
            "nodes": len(nodes),
            "ontology_disease_nodes": split["ontology_disease_count"],
            "sampled_test_pairs": split["sampled_test_pair_count"],
            "directed_edges": split["directed_edge_count"],
            "final_test_diseases": disease_count,
            **{name: len(frame) for name, frame in parts.items()},
        },
        "files": {},
    }

    try:
        _write(temporary, nodes, parts, manifest)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            shutil.rmtree(path)

        temporary.replace(path)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return path
