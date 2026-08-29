__all__ = (
    "BENCHMARK_SOURCE_DIGEST",
    "BENCHMARK_VERSION",
    "DEFAULT_BENCHMARK_ROOT",
    "BenchmarkReport",
    "build_benchmark",
    "validate_benchmark",
)

import json
import platform
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import polars as pl

from .audit import ADVERSE_RELATIONS, THERAPEUTIC_RELATIONS
from .source import EDGE_FILES, NODE_FILES, SourceSnapshot

BENCHMARK_VERSION = "optimuskg-v1"
BENCHMARK_SOURCE_DIGEST = "b4d46b4195d59b827354a382723ca8a73b66afe06869e6ff9167d539819754ea"
DEFAULT_BENCHMARK_ROOT = Path("data/benchmark") / BENCHMARK_VERSION

PRIMARY_EXCLUDED_EDGE_LABELS = ("DRG-PHE",)
PRIMARY_EXCLUDED_RELATIONS = (*THERAPEUTIC_RELATIONS, *ADVERSE_RELATIONS)
PRIMARY_EXCLUDED_EDGE_TYPES = (
    ("DIS-DIS", "PARENT"),
    ("DRG-DRG", "PARENT"),
)

SPLIT_SALT = "drepurpose:optimuskg-v1"
FREQUENCY_BANDS = (
    "1",
    "2-3",
    "4-7",
    "8-15",
    "16-31",
    "32-63",
    "64-127",
    "128+",
)

_ARTIFACT_FILES = (
    "nodes.parquet",
    "edges.parquet",
    "drugs.parquet",
    "diseases.parquet",
    "excluded_diseases.parquet",
    "indications/train.parquet",
    "indications/validation.parquet",
    "indications/test.parquet",
)
_PARQUET_ROW_GROUP_SIZE = 100_000
_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    root: Path
    nodes: int
    edges: int
    drugs: int
    diseases: int
    excluded_diseases: int
    train_diseases: int
    validation_diseases: int
    test_diseases: int
    train_indications: int
    validation_indications: int
    test_indications: int


def _property(name: str) -> pl.Expr:
    return pl.col("properties").struct.field(name)


def _nonempty(expression: pl.Expr) -> pl.Expr:
    return expression.is_not_null() & (expression != "")


def _sha256_file(path: Path) -> str:
    digest = sha256()

    with path.open("rb") as file:
        while chunk := file.read(_CHUNK_SIZE):
            digest.update(chunk)

    return digest.hexdigest()


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _git_state() -> tuple[str | None, bool | None]:
    root = _repository_root()

    try:
        revision = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ("git", "status", "--porcelain"),
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except FileNotFoundError, subprocess.CalledProcessError:
        return None, None

    return revision, dirty


def _lock_digest() -> str | None:
    path = _repository_root() / "uv.lock"
    return _sha256_file(path) if path.is_file() else None


def _source_digest(snapshot: SourceSnapshot) -> str:
    digest = sha256()

    for file in sorted(snapshot.files, key=lambda file: file.relative_path):
        digest.update(file.relative_path.encode())
        digest.update(b"\0")
        digest.update(file.sha256.encode())
        digest.update(b"\n")

    return digest.hexdigest()


def _validate_source(snapshot: SourceSnapshot) -> str:
    digest = _source_digest(snapshot)

    if digest != BENCHMARK_SOURCE_DIGEST:
        raise RuntimeError(
            "Source snapshot does not match the frozen OptimusKG snapshot for "
            f"{BENCHMARK_VERSION}: expected {BENCHMARK_SOURCE_DIGEST}, got {digest}"
        )

    return digest


def _row_count(path: Path) -> int:
    return int(pl.scan_parquet(path).select(pl.len()).collect().item())


def _write_parquet(frame: pl.LazyFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.sink_parquet(
        path,
        compression="zstd",
        compression_level=3,
        row_group_size=_PARQUET_ROW_GROUP_SIZE,
        maintain_order=True,
    )


def _schema_valid_edges(nodes: pl.LazyFrame, edges: pl.LazyFrame) -> pl.LazyFrame:
    labels = nodes.select("id", pl.col("label").alias("node_label"))

    typed = (
        edges.select("from", "to", "label", "relation", "undirected")
        .join(
            labels,
            left_on="from",
            right_on="id",
            how="left",
            maintain_order="left",
        )
        .rename({"node_label": "from_label"})
        .join(
            labels,
            left_on="to",
            right_on="id",
            how="left",
            maintain_order="left",
        )
        .rename({"node_label": "to_label"})
    )

    parts = pl.col("label").str.split_exact("-", 1)
    expected_from = parts.struct.field("field_0")
    expected_to = parts.struct.field("field_1")

    forward = (pl.col("from_label") == expected_from) & (pl.col("to_label") == expected_to)
    reverse = (pl.col("from_label") == expected_to) & (pl.col("to_label") == expected_from)
    valid = pl.when(pl.col("undirected")).then(forward | reverse).otherwise(forward)

    return typed.filter(valid).select("from", "to", "label", "relation", "undirected")


def _primary_edges(nodes: pl.LazyFrame, edges: pl.LazyFrame) -> pl.LazyFrame:
    excluded_type = pl.any_horizontal(
        *(
            (pl.col("label") == label) & (pl.col("relation") == relation)
            for label, relation in PRIMARY_EXCLUDED_EDGE_TYPES
        )
    )

    return _schema_valid_edges(nodes, edges).filter(
        ~pl.col("label").is_in(PRIMARY_EXCLUDED_EDGE_LABELS),
        ~pl.col("relation").is_in(PRIMARY_EXCLUDED_RELATIONS),
        ~excluded_type,
    )


def _background_degree(edges: pl.LazyFrame, ids: pl.LazyFrame) -> pl.LazyFrame:
    incident = pl.concat(
        (
            edges.select(pl.col("from").alias("id")),
            edges.select(pl.col("to").alias("id")),
        )
    )
    degree = incident.group_by("id").agg(pl.len().alias("background_degree"))

    return (
        ids.join(degree, on="id", how="left")
        .with_columns(pl.col("background_degree").fill_null(0).cast(pl.UInt32))
        .sort("id")
    )


def _identity_identifiers(frame: pl.LazyFrame) -> pl.LazyFrame:
    primary = frame.select(
        "id",
        _property("umls_cui").alias("identifier"),
    )
    concepts = frame.select(
        "id",
        _property("concept_ids").alias("identifier"),
    ).explode("identifier")

    return pl.concat((primary, concepts)).filter(_nonempty(pl.col("identifier"))).unique()


def _identity_exclusions(
    targets: pl.LazyFrame,
    diseases: pl.LazyFrame,
    phenotypes: pl.LazyFrame,
) -> pl.LazyFrame:
    disease_identifiers = _identity_identifiers(diseases)
    phenotype_identifiers = _identity_identifiers(phenotypes)

    duplicate_disease_identifiers = (
        disease_identifiers.group_by("identifier")
        .agg(pl.col("id").n_unique().alias("diseases"))
        .filter(pl.col("diseases") > 1)
        .select("identifier")
    )

    duplicate_disease_identity = (
        disease_identifiers.join(duplicate_disease_identifiers, on="identifier", how="inner")
        .join(targets.select("id"), on="id", how="inner")
        .select(
            "id",
            pl.lit("duplicate_disease_identity_identifier").alias("reason"),
            "identifier",
        )
    )

    disease_phenotype_overlap = (
        disease_identifiers.join(
            phenotype_identifiers.select("identifier").unique(),
            on="identifier",
            how="inner",
        )
        .join(targets.select("id"), on="id", how="inner")
        .select(
            "id",
            pl.lit("disease_phenotype_identity_identifier_overlap").alias("reason"),
            "identifier",
        )
    )

    return pl.concat((duplicate_disease_identity, disease_phenotype_overlap)).unique()


def _frequency_band(count: int) -> str:
    if count == 1:
        return "1"
    if count <= 3:
        return "2-3"
    if count <= 7:
        return "4-7"
    if count <= 15:
        return "8-15"
    if count <= 31:
        return "16-31"
    if count <= 63:
        return "32-63"
    if count <= 127:
        return "64-127"
    return "128+"


def _split_order(disease_id: str) -> bytes:
    return sha256(f"{SPLIT_SALT}:{disease_id}".encode()).digest()


def _split_diseases(diseases: pl.DataFrame) -> pl.DataFrame:
    groups: dict[str, list[tuple[str, int, int]]] = {band: [] for band in FREQUENCY_BANDS}

    for disease_id, indication_count, background_degree in diseases.select(
        "id",
        "indication_count",
        "background_degree",
    ).iter_rows():
        band = _frequency_band(int(indication_count))
        groups[band].append((str(disease_id), int(indication_count), int(background_degree)))

    rows: list[tuple[str, str, int, int, str]] = []

    for band in FREQUENCY_BANDS:
        group = sorted(groups[band], key=lambda row: (_split_order(row[0]), row[0]))
        validation_size = len(group) // 10
        test_size = len(group) // 10
        train_size = len(group) - validation_size - test_size

        for index, (disease_id, indication_count, background_degree) in enumerate(group):
            if index < train_size:
                split = "train"
            elif index < train_size + validation_size:
                split = "validation"
            else:
                split = "test"

            rows.append((disease_id, split, indication_count, background_degree, band))

    return pl.DataFrame(
        rows,
        schema={
            "id": pl.String,
            "split": pl.String,
            "indication_count": pl.UInt32,
            "background_degree": pl.UInt32,
            "frequency_band": pl.String,
        },
        orient="row",
    ).sort("id")


def _artifact_info(root: Path) -> dict[str, dict[str, str | int]]:
    return {
        relative_path: {
            "rows": _row_count(root / relative_path),
            "size": (root / relative_path).stat().st_size,
            "sha256": _sha256_file(root / relative_path),
        }
        for relative_path in _ARTIFACT_FILES
    }


def _counts(frame: pl.DataFrame, column: str) -> dict[str, int]:
    return {
        str(value): int(count)
        for value, count in frame.group_by(column).len().sort(column).iter_rows()
    }


def _write_manifest(
    root: Path,
    snapshot: SourceSnapshot,
    source_digest: str,
    diseases: pl.DataFrame,
    excluded: pl.DataFrame,
    drugs: pl.DataFrame,
) -> None:
    artifacts = _artifact_info(root)
    revision, dirty = _git_state()

    manifest = {
        "benchmark": BENCHMARK_VERSION,
        "source": {
            "doi": snapshot.doi,
            "digest": source_digest,
            "files": {
                file.relative_path: {
                    "size": file.size,
                    "sha256": file.sha256,
                }
                for file in snapshot.files
            },
        },
        "policy": {
            "target": {
                "label": "DRG-DIS",
                "relation": "INDICATION",
            },
            "candidate_drugs": "all DRG nodes",
            "reasoning_graph": {
                "source": "full OptimusKG graph",
                "remove_schema_invalid_edges": True,
                "excluded_edge_labels": list(PRIMARY_EXCLUDED_EDGE_LABELS),
                "excluded_relations": list(PRIMARY_EXCLUDED_RELATIONS),
                "excluded_edge_types": [
                    list(edge_type) for edge_type in PRIMARY_EXCLUDED_EDGE_TYPES
                ],
            },
            "disease_eligibility": {
                "requires_indication": True,
                "minimum_background_degree": 1,
                "exclude_duplicate_disease_identity_identifiers": True,
                "exclude_disease_phenotype_identity_identifier_overlap": True,
                "identity_fields": ["umls_cui", "concept_ids"],
            },
            "split": {
                "unit": "disease",
                "frequency_bands": list(FREQUENCY_BANDS),
                "validation_per_band": "floor(n / 10)",
                "test_per_band": "floor(n / 10)",
                "train_per_band": "remainder",
                "ordering": "SHA-256",
                "salt": SPLIT_SALT,
            },
        },
        "counts": {
            "nodes": artifacts["nodes.parquet"]["rows"],
            "reasoning_edges": artifacts["edges.parquet"]["rows"],
            "candidate_drugs": artifacts["drugs.parquet"]["rows"],
            "candidate_drugs_without_background": int(
                drugs.filter(pl.col("background_degree") == 0).height
            ),
            "eligible_diseases": diseases.height,
            "excluded_diseases": (
                int(excluded.get_column("id").n_unique()) if excluded.height else 0
            ),
            "diseases_by_split": _counts(diseases, "split"),
            "exclusions_by_reason": _counts(excluded, "reason") if excluded.height else {},
            "indications_by_split": {
                "train": artifacts["indications/train.parquet"]["rows"],
                "validation": artifacts["indications/validation.parquet"]["rows"],
                "test": artifacts["indications/test.parquet"]["rows"],
            },
        },
        "software": {
            "python": platform.python_version(),
            "polars": pl.__version__,
            "git_revision": revision,
            "git_dirty": dirty,
            "uv_lock_sha256": _lock_digest(),
        },
        "artifacts": artifacts,
    }

    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def validate_benchmark(root: Path) -> None:
    missing = [path for path in (*_ARTIFACT_FILES, "manifest.json") if not (root / path).is_file()]

    if missing:
        raise RuntimeError("Incomplete benchmark: " + ", ".join(missing))

    manifest = json.loads((root / "manifest.json").read_text())
    edges = pl.scan_parquet(root / "edges.parquet")
    nodes = pl.scan_parquet(root / "nodes.parquet")
    drugs = pl.read_parquet(root / "drugs.parquet")
    diseases = pl.read_parquet(root / "diseases.parquet")

    failures: list[str] = []

    if manifest.get("benchmark") != BENCHMARK_VERSION:
        failures.append("manifest benchmark version mismatch")

    if manifest.get("source", {}).get("digest") != BENCHMARK_SOURCE_DIGEST:
        failures.append("manifest source digest mismatch")

    for relative_path, info in manifest.get("artifacts", {}).items():
        path = root / relative_path

        if not path.is_file():
            failures.append(f"missing artifact: {relative_path}")
            continue

        if _sha256_file(path) != info.get("sha256"):
            failures.append(f"artifact hash mismatch: {relative_path}")

    excluded_type = pl.any_horizontal(
        *(
            (pl.col("label") == label) & (pl.col("relation") == relation)
            for label, relation in PRIMARY_EXCLUDED_EDGE_TYPES
        )
    )
    invalid_edges = edges.filter(
        pl.col("label").is_in(PRIMARY_EXCLUDED_EDGE_LABELS)
        | pl.col("relation").is_in(PRIMARY_EXCLUDED_RELATIONS)
        | excluded_type
    )

    if _row_count(root / "edges.parquet") == 0:
        failures.append("reasoning graph is empty")

    if int(invalid_edges.select(pl.len()).collect().item()):
        failures.append("reasoning graph contains excluded edges")

    candidate_ids = set(drugs.get_column("id").to_list())
    graph_drugs = set(
        nodes.filter(pl.col("label") == "DRG").select("id").collect().get_column("id").to_list()
    )

    if candidate_ids != graph_drugs:
        failures.append("candidate drugs do not match DRG nodes")

    if diseases.filter(pl.col("background_degree") == 0).height:
        failures.append("eligible diseases without background context")

    if set(diseases.get_column("split")) != {"train", "validation", "test"}:
        failures.append("benchmark does not contain all three disease splits")

    disease_split = dict(diseases.select("id", "split").iter_rows())
    seen_pairs: set[tuple[str, str]] = set()

    for split in ("train", "validation", "test"):
        labels = pl.read_parquet(root / f"indications/{split}.parquet")

        for drug_id, disease_id in labels.select("drug_id", "disease_id").iter_rows():
            pair = (str(drug_id), str(disease_id))

            if pair in seen_pairs:
                failures.append(f"duplicate indication pair across splits: {pair}")
                break

            seen_pairs.add(pair)

            if drug_id not in candidate_ids:
                failures.append(f"non-candidate indication drug: {drug_id}")
                break

            if disease_split.get(str(disease_id)) != split:
                failures.append(f"indication disease assigned to wrong split: {disease_id}")
                break

    if failures:
        details = "\n".join(f"- {failure}" for failure in failures)
        raise RuntimeError(f"Benchmark validation failed:\n{details}")


def build_benchmark(
    snapshot: SourceSnapshot,
    output: Path = DEFAULT_BENCHMARK_ROOT,
) -> BenchmarkReport:
    source_digest = _validate_source(snapshot)
    output = output.expanduser().resolve()

    if output.exists():
        raise FileExistsError(f"Benchmark output already exists: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))

    try:
        source_nodes = pl.scan_parquet(snapshot.path("nodes.parquet"))
        source_edges = pl.scan_parquet(snapshot.path("edges.parquet"))

        _write_parquet(source_nodes.select("id", "label"), temp / "nodes.parquet")
        _write_parquet(_primary_edges(source_nodes, source_edges), temp / "edges.parquet")

        reasoning_edges = pl.scan_parquet(temp / "edges.parquet")

        drug_ids = source_nodes.filter(pl.col("label") == "DRG").select("id")
        drugs = _background_degree(reasoning_edges, drug_ids).collect()
        _write_parquet(drugs.lazy(), temp / "drugs.parquet")

        drug_disease = pl.scan_parquet(snapshot.path(EDGE_FILES["DRG-DIS"]))
        indications = drug_disease.filter(pl.col("relation") == "INDICATION").select(
            pl.col("from").alias("drug_id"),
            pl.col("to").alias("disease_id"),
        )

        targets = (
            indications.group_by("disease_id")
            .agg(pl.len().cast(pl.UInt32).alias("indication_count"))
            .rename({"disease_id": "id"})
        )
        target_degrees = _background_degree(reasoning_edges, targets.select("id")).join(
            targets,
            on="id",
            how="inner",
        )

        disease_nodes = pl.scan_parquet(snapshot.path(NODE_FILES["DIS"]))
        phenotype_nodes = pl.scan_parquet(snapshot.path(NODE_FILES["PHE"]))
        identity_exclusions = _identity_exclusions(targets, disease_nodes, phenotype_nodes)
        no_background = target_degrees.filter(pl.col("background_degree") == 0).select(
            "id",
            pl.lit("no_background_context").alias("reason"),
            pl.lit(None, dtype=pl.String).alias("identifier"),
        )

        excluded = (
            pl.concat((identity_exclusions, no_background))
            .unique()
            .sort("id", "reason", "identifier")
            .collect()
        )
        _write_parquet(excluded.lazy(), temp / "excluded_diseases.parquet")

        eligible = (
            target_degrees.join(
                excluded.lazy().select("id").unique(),
                on="id",
                how="anti",
            )
            .select("id", "indication_count", "background_degree")
            .collect()
        )
        diseases = _split_diseases(eligible)
        _write_parquet(diseases.lazy(), temp / "diseases.parquet")

        disease_splits = diseases.select("id", "split").lazy()
        split_indications = indications.join(
            disease_splits,
            left_on="disease_id",
            right_on="id",
            how="inner",
        )

        for split in ("train", "validation", "test"):
            _write_parquet(
                split_indications.filter(pl.col("split") == split)
                .select("drug_id", "disease_id")
                .sort("disease_id", "drug_id"),
                temp / f"indications/{split}.parquet",
            )

        _write_manifest(temp, snapshot, source_digest, diseases, excluded, drugs)
        validate_benchmark(temp)
        temp.replace(output)
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise

    manifest = json.loads((output / "manifest.json").read_text())
    counts = manifest["counts"]
    disease_counts = counts["diseases_by_split"]
    indication_counts = counts["indications_by_split"]

    return BenchmarkReport(
        root=output,
        nodes=int(counts["nodes"]),
        edges=int(counts["reasoning_edges"]),
        drugs=int(counts["candidate_drugs"]),
        diseases=int(counts["eligible_diseases"]),
        excluded_diseases=int(counts["excluded_diseases"]),
        train_diseases=int(disease_counts.get("train", 0)),
        validation_diseases=int(disease_counts.get("validation", 0)),
        test_diseases=int(disease_counts.get("test", 0)),
        train_indications=int(indication_counts["train"]),
        validation_indications=int(indication_counts["validation"]),
        test_indications=int(indication_counts["test"]),
    )
