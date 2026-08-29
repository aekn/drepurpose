__all__ = (
    "ADVERSE_RELATIONS",
    "THERAPEUTIC_RELATIONS",
    "AuditReport",
    "audit_source",
    "validate_audit",
    "write_audit",
)

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import polars as pl

from .source import EDGE_FILES, NODE_FILES, SourceSnapshot

THERAPEUTIC_RELATIONS = (
    "INDICATION",
    "CONTRAINDICATION",
    "OFF_LABEL_USE",
)

ADVERSE_RELATIONS = ("ADVERSE_DRUG_REACTION",)

_NODE_COLUMNS = ("id", "label", "properties")
_EDGE_COLUMNS = ("from", "to", "label", "relation", "undirected", "properties")


@dataclass(frozen=True, slots=True)
class NumericSummary:
    non_null: int
    null: int
    minimum: float | None
    median: float | None
    p90: float | None
    maximum: float | None


@dataclass(frozen=True, slots=True)
class CountDistribution:
    minimum: int
    median: float
    p90: float
    p99: float
    maximum: int
    histogram: dict[str, int]


@dataclass(frozen=True, slots=True)
class ComponentAudit:
    edges: int
    nodes: int
    components: int
    largest_component: int
    size_histogram: dict[str, int]


@dataclass(frozen=True, slots=True)
class ProvenanceAudit:
    direct: dict[str, int]
    indirect: dict[str, int]


@dataclass(frozen=True, slots=True)
class _EndpointAudit:
    orphan_source_edges: int
    orphan_target_edges: int
    endpoint_type_mismatches: int
    endpoint_type_mismatch_counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class IndicationAudit:
    edges: int
    diseases: int
    drugs: int
    per_disease: CountDistribution
    per_drug: CountDistribution
    clinical_trial_phase: dict[str, int]


@dataclass(frozen=True, slots=True)
class IdentityAudit:
    disease_primary_cui_duplicate_groups: int
    disease_primary_cui_duplicate_nodes: int
    disease_phenotype_shared_primary_cuis: int
    disease_phenotype_shared_concept_ids: int


@dataclass(frozen=True, slots=True)
class DrugAudit:
    approval: dict[str, int]
    status: dict[str, int]
    indication_drug_approval: dict[str, int]
    parent_graph: ComponentAudit


@dataclass(frozen=True, slots=True)
class AuditReport:
    dataset_doi: str
    files: dict[str, dict[str, str | int]]
    source_size: int
    node_count: int
    edge_count: int
    largest_connected_component_nodes: int
    largest_connected_component_edges: int
    node_types: dict[str, int]
    edge_types: dict[str, int]
    edge_relations: dict[str, int]
    edge_directionality: dict[str, int]
    null_node_ids: int
    null_edge_fields: int
    duplicate_node_ids: int
    duplicate_edge_keys: int
    orphan_source_edges: int
    orphan_target_edges: int
    endpoint_type_mismatches: int
    endpoint_type_mismatch_counts: dict[str, int]
    directionality_conflicts: int
    self_loops: int
    typed_view_matches: dict[str, bool]
    therapeutic_edges: dict[str, int]
    adverse_edges: dict[str, int]
    drug_disease_conflict_pairs: int
    indications: IndicationAudit
    identity: IdentityAudit
    drugs: DrugAudit
    disease_gene_evidence: dict[str, NumericSummary]
    anatomy_gene_relations: dict[str, int]
    anatomy_gene_quality: dict[str, int]
    edge_provenance: dict[str, ProvenanceAudit]


def _scan(path: Path) -> pl.LazyFrame:
    return pl.scan_parquet(path)


def _require_columns(frame: pl.LazyFrame, path: Path, required: tuple[str, ...]) -> None:
    columns = set(frame.collect_schema().names())
    missing = set(required) - columns

    if missing:
        names = ", ".join(sorted(missing))
        raise RuntimeError(f"{path} is missing required columns: {names}")


def _count(frame: pl.LazyFrame) -> int:
    return int(frame.select(pl.len()).collect().item())


def _counts(frame: pl.LazyFrame, *columns: str) -> dict[str, int]:
    rows = frame.group_by(*columns).len().sort(*columns).collect()
    counts: dict[str, int] = {}

    for row in rows.iter_rows():
        values = row[:-1]
        count = row[-1]
        key = "/".join("null" if value is None else str(value) for value in values)
        counts[key] = int(count)

    return counts


def _value_counts(frame: pl.LazyFrame, expression: pl.Expr) -> dict[str, int]:
    return _counts(frame.select(expression.alias("value")), "value")


def _property(name: str) -> pl.Expr:
    return pl.col("properties").struct.field(name)


def _nonempty_string(expression: pl.Expr) -> pl.Expr:
    return expression.is_not_null() & (expression != "")


def _count_distribution(frame: pl.LazyFrame, column: str) -> CountDistribution:
    summary = frame.select(
        pl.col(column).min().alias("minimum"),
        pl.col(column).median().alias("median"),
        pl.col(column).quantile(0.90).alias("p90"),
        pl.col(column).quantile(0.99).alias("p99"),
        pl.col(column).max().alias("maximum"),
    ).collect()

    histogram = frame.group_by(column).len().sort(column).collect()

    return CountDistribution(
        minimum=int(summary["minimum"].item()),
        median=float(summary["median"].item()),
        p90=float(summary["p90"].item()),
        p99=float(summary["p99"].item()),
        maximum=int(summary["maximum"].item()),
        histogram={str(value): int(count) for value, count in histogram.iter_rows()},
    )


def _numeric_summaries(
    frame: pl.LazyFrame,
    fields: tuple[str, ...],
) -> dict[str, NumericSummary]:
    expressions: list[pl.Expr] = []

    for field in fields:
        value = _property(field)
        expressions.extend(
            (
                value.count().alias(f"{field}__non_null"),
                value.is_null().sum().alias(f"{field}__null"),
                value.min().alias(f"{field}__minimum"),
                value.median().alias(f"{field}__median"),
                value.quantile(0.90).alias(f"{field}__p90"),
                value.max().alias(f"{field}__maximum"),
            )
        )

    summary = frame.select(expressions).collect()
    result: dict[str, NumericSummary] = {}

    for field in fields:
        minimum = summary[f"{field}__minimum"].item()
        median = summary[f"{field}__median"].item()
        p90 = summary[f"{field}__p90"].item()
        maximum = summary[f"{field}__maximum"].item()

        result[field] = NumericSummary(
            non_null=int(summary[f"{field}__non_null"].item()),
            null=int(summary[f"{field}__null"].item()),
            minimum=None if minimum is None else float(minimum),
            median=None if median is None else float(median),
            p90=None if p90 is None else float(p90),
            maximum=None if maximum is None else float(maximum),
        )

    return result


def _duplicate_edge_keys(edges: pl.LazyFrame) -> int:
    keys = edges.select(
        pl.when(pl.col("undirected") & (pl.col("from") > pl.col("to")))
        .then(pl.col("to"))
        .otherwise(pl.col("from"))
        .alias("left"),
        pl.when(pl.col("undirected") & (pl.col("from") > pl.col("to")))
        .then(pl.col("from"))
        .otherwise(pl.col("to"))
        .alias("right"),
        "label",
        "relation",
        "undirected",
    )

    return int(
        keys.group_by("left", "right", "label", "relation", "undirected")
        .len()
        .filter(pl.col("len") > 1)
        .select((pl.col("len") - 1).sum().fill_null(0))
        .collect()
        .item()
    )


def _audit_endpoints(
    nodes: pl.LazyFrame,
    edges: pl.LazyFrame,
) -> _EndpointAudit:
    node_labels = nodes.select(
        "id",
        pl.col("label").alias("node_label"),
    )

    typed_edges = (
        edges.select("from", "to", "label", "relation", "undirected")
        .join(node_labels, left_on="from", right_on="id", how="left")
        .rename({"node_label": "from_label"})
        .join(node_labels, left_on="to", right_on="id", how="left")
        .rename({"node_label": "to_label"})
    )

    label_parts = pl.col("label").str.split_exact("-", 1)
    expected_from = label_parts.struct.field("field_0")
    expected_to = label_parts.struct.field("field_1")

    forward = (pl.col("from_label") == expected_from) & (pl.col("to_label") == expected_to)
    reverse = (pl.col("from_label") == expected_to) & (pl.col("to_label") == expected_from)
    valid = pl.when(pl.col("undirected")).then(forward | reverse).otherwise(forward)

    mismatches = typed_edges.filter(
        pl.col("from_label").is_not_null() & pl.col("to_label").is_not_null() & ~valid
    )

    summary = typed_edges.select(
        pl.col("from_label").is_null().sum().alias("orphan_sources"),
        pl.col("to_label").is_null().sum().alias("orphan_targets"),
    ).collect()

    return _EndpointAudit(
        orphan_source_edges=int(summary["orphan_sources"].item()),
        orphan_target_edges=int(summary["orphan_targets"].item()),
        endpoint_type_mismatches=_count(mismatches),
        endpoint_type_mismatch_counts=_counts(
            mismatches,
            "label",
            "relation",
            "from_label",
            "to_label",
        ),
    )


def _typed_view_matches(
    snapshot: SourceSnapshot,
    node_types: dict[str, int],
    edge_types: dict[str, int],
) -> dict[str, bool]:
    matches: dict[str, bool] = {}

    for label, path in NODE_FILES.items():
        frame = _scan(snapshot.path(path))
        _require_columns(frame, snapshot.path(path), _NODE_COLUMNS)

        label_matches = bool(frame.select((pl.col("label") == label).all()).collect().item())
        matches[path] = _count(frame) == node_types.get(label, 0) and label_matches

    for label, path in EDGE_FILES.items():
        frame = _scan(snapshot.path(path))
        _require_columns(frame, snapshot.path(path), _EDGE_COLUMNS)

        label_matches = bool(frame.select((pl.col("label") == label).all()).collect().item())
        matches[path] = _count(frame) == edge_types.get(label, 0) and label_matches

    return matches


def _audit_identity(
    diseases: pl.LazyFrame,
    phenotypes: pl.LazyFrame,
) -> IdentityAudit:
    disease_cuis = diseases.select(
        "id",
        _property("umls_cui").alias("cui"),
    ).filter(_nonempty_string(pl.col("cui")))

    phenotype_cuis = phenotypes.select(
        "id",
        _property("umls_cui").alias("cui"),
    ).filter(_nonempty_string(pl.col("cui")))

    duplicate_groups = (
        disease_cuis.group_by("cui")
        .agg(pl.col("id").n_unique().alias("nodes"))
        .filter(pl.col("nodes") > 1)
    )

    disease_primary_cui_duplicate_groups = _count(duplicate_groups)
    disease_primary_cui_duplicate_nodes = int(
        duplicate_groups.select(pl.col("nodes").sum().fill_null(0)).collect().item()
    )

    disease_phenotype_shared_primary_cuis = _count(
        disease_cuis.select("cui")
        .unique()
        .join(
            phenotype_cuis.select("cui").unique(),
            on="cui",
            how="inner",
        )
    )

    disease_concepts = (
        diseases.select(_property("concept_ids").alias("concept_id"))
        .explode("concept_id")
        .filter(_nonempty_string(pl.col("concept_id")))
        .unique()
    )
    phenotype_concepts = (
        phenotypes.select(_property("concept_ids").alias("concept_id"))
        .explode("concept_id")
        .filter(_nonempty_string(pl.col("concept_id")))
        .unique()
    )

    disease_phenotype_shared_concept_ids = _count(
        disease_concepts.join(
            phenotype_concepts,
            on="concept_id",
            how="inner",
        )
    )

    return IdentityAudit(
        disease_primary_cui_duplicate_groups=disease_primary_cui_duplicate_groups,
        disease_primary_cui_duplicate_nodes=disease_primary_cui_duplicate_nodes,
        disease_phenotype_shared_primary_cuis=disease_phenotype_shared_primary_cuis,
        disease_phenotype_shared_concept_ids=disease_phenotype_shared_concept_ids,
    )


def _component_audit(frame: pl.LazyFrame) -> ComponentAudit:
    pairs = frame.select("from", "to").collect()

    if pairs.is_empty():
        return ComponentAudit(
            edges=0,
            nodes=0,
            components=0,
            largest_component=0,
            size_histogram={},
        )

    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)

        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]

        return node

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)

        if left_root != right_root:
            parent[right_root] = left_root

    for left, right in pairs.iter_rows():
        union(str(left), str(right))

    sizes = Counter(find(node) for node in parent)
    histogram = Counter(sizes.values())

    return ComponentAudit(
        edges=pairs.height,
        nodes=len(parent),
        components=len(sizes),
        largest_component=max(sizes.values()),
        size_histogram={str(size): count for size, count in sorted(histogram.items())},
    )


def _audit_drugs(
    drugs: pl.LazyFrame,
    indications: pl.LazyFrame,
    drug_drug: pl.LazyFrame,
) -> DrugAudit:
    drug_info = drugs.select(
        "id",
        _property("is_approved").alias("is_approved"),
        _property("status").alias("status"),
    )

    indication_drugs = indications.select(pl.col("from").alias("id")).unique()
    indication_drug_info = drug_info.join(indication_drugs, on="id", how="inner")
    parent_edges = drug_drug.filter(pl.col("relation") == "PARENT")

    return DrugAudit(
        approval=_value_counts(drug_info, pl.col("is_approved")),
        status=_value_counts(drug_info, pl.col("status")),
        indication_drug_approval=_value_counts(
            indication_drug_info,
            pl.col("is_approved"),
        ),
        parent_graph=_component_audit(parent_edges),
    )


def _audit_indications(drug_disease: pl.LazyFrame) -> IndicationAudit:
    indications = drug_disease.filter(pl.col("relation") == "INDICATION")

    if _count(indications) == 0:
        raise RuntimeError("OptimusKG contains no DRG-DIS/INDICATION edges")

    per_disease = indications.group_by("to").agg(pl.len().alias("count"))
    per_drug = indications.group_by("from").agg(pl.len().alias("count"))

    return IndicationAudit(
        edges=_count(indications),
        diseases=_count(indications.select("to").unique()),
        drugs=_count(indications.select("from").unique()),
        per_disease=_count_distribution(per_disease, "count"),
        per_drug=_count_distribution(per_drug, "count"),
        clinical_trial_phase=_value_counts(
            indications,
            _property("highest_clinical_trial_phase"),
        ),
    )


def _provenance(frame: pl.LazyFrame) -> ProvenanceAudit:
    sources = _property("sources")

    direct = (
        frame.select(sources.struct.field("direct").alias("source"))
        .explode("source")
        .filter(_nonempty_string(pl.col("source")))
    )
    indirect = (
        frame.select(sources.struct.field("indirect").alias("source"))
        .explode("source")
        .filter(_nonempty_string(pl.col("source")))
    )

    return ProvenanceAudit(
        direct=_counts(direct, "source"),
        indirect=_counts(indirect, "source"),
    )


def _edge_provenance(snapshot: SourceSnapshot) -> dict[str, ProvenanceAudit]:
    return {label: _provenance(_scan(snapshot.path(path))) for label, path in EDGE_FILES.items()}


def audit_source(snapshot: SourceSnapshot) -> AuditReport:
    nodes = _scan(snapshot.path("nodes.parquet"))
    edges = _scan(snapshot.path("edges.parquet"))
    lcc_nodes = _scan(snapshot.path("largest_connected_component_nodes.parquet"))
    lcc_edges = _scan(snapshot.path("largest_connected_component_edges.parquet"))

    _require_columns(nodes, snapshot.path("nodes.parquet"), _NODE_COLUMNS)
    _require_columns(edges, snapshot.path("edges.parquet"), _EDGE_COLUMNS)
    _require_columns(
        lcc_nodes,
        snapshot.path("largest_connected_component_nodes.parquet"),
        _NODE_COLUMNS,
    )
    _require_columns(
        lcc_edges,
        snapshot.path("largest_connected_component_edges.parquet"),
        _EDGE_COLUMNS,
    )

    node_types = _counts(nodes, "label")
    edge_types = _counts(edges, "label")

    endpoints = _audit_endpoints(nodes, edges)

    directionality_conflicts = _count(
        edges.group_by("label", "relation")
        .agg(pl.col("undirected").n_unique().alias("values"))
        .filter(pl.col("values") > 1)
    )

    therapeutic = edges.filter(pl.col("relation").is_in(THERAPEUTIC_RELATIONS))
    adverse = edges.filter(pl.col("relation").is_in(ADVERSE_RELATIONS))

    drugs = _scan(snapshot.path(NODE_FILES["DRG"]))
    diseases = _scan(snapshot.path(NODE_FILES["DIS"]))
    phenotypes = _scan(snapshot.path(NODE_FILES["PHE"]))
    drug_disease = _scan(snapshot.path(EDGE_FILES["DRG-DIS"]))
    drug_drug = _scan(snapshot.path(EDGE_FILES["DRG-DRG"]))
    disease_gene = _scan(snapshot.path(EDGE_FILES["DIS-GEN"]))
    anatomy_gene = _scan(snapshot.path(EDGE_FILES["ANA-GEN"]))

    indications = drug_disease.filter(pl.col("relation") == "INDICATION")

    drug_disease_conflict_pairs = _count(
        drug_disease.filter(pl.col("relation").is_in(THERAPEUTIC_RELATIONS))
        .group_by("from", "to")
        .agg(pl.col("relation").n_unique().alias("relations"))
        .filter(pl.col("relations") > 1)
    )

    evidence_fields = (
        "evidence_score",
        "evidence_count",
        "evidence_index",
        "disease_specificity_index",
        "disease_pleiotropy_index",
        "disgenet_score",
        "number_of_pmids",
        "number_of_snps",
    )

    files = {
        file.relative_path: {
            "size": file.size,
            "sha256": file.sha256,
        }
        for file in snapshot.files
    }

    return AuditReport(
        dataset_doi=snapshot.doi,
        files=files,
        source_size=sum(file.size for file in snapshot.files),
        node_count=_count(nodes),
        edge_count=_count(edges),
        largest_connected_component_nodes=_count(lcc_nodes),
        largest_connected_component_edges=_count(lcc_edges),
        node_types=node_types,
        edge_types=edge_types,
        edge_relations=_counts(edges, "label", "relation"),
        edge_directionality=_counts(edges, "label", "relation", "undirected"),
        null_node_ids=int(nodes.select(pl.col("id").is_null().sum()).collect().item()),
        null_edge_fields=int(
            edges.select(
                (
                    pl.col("from").is_null()
                    | pl.col("to").is_null()
                    | pl.col("label").is_null()
                    | pl.col("relation").is_null()
                    | pl.col("undirected").is_null()
                )
                .sum()
                .alias("count")
            )
            .collect()
            .item()
        ),
        duplicate_node_ids=int(
            nodes.group_by("id")
            .len()
            .filter(pl.col("len") > 1)
            .select((pl.col("len") - 1).sum().fill_null(0))
            .collect()
            .item()
        ),
        duplicate_edge_keys=_duplicate_edge_keys(edges),
        orphan_source_edges=endpoints.orphan_source_edges,
        orphan_target_edges=endpoints.orphan_target_edges,
        endpoint_type_mismatches=endpoints.endpoint_type_mismatches,
        endpoint_type_mismatch_counts=endpoints.endpoint_type_mismatch_counts,
        directionality_conflicts=directionality_conflicts,
        self_loops=_count(edges.filter(pl.col("from") == pl.col("to"))),
        typed_view_matches=_typed_view_matches(snapshot, node_types, edge_types),
        therapeutic_edges=_counts(therapeutic, "label", "relation"),
        adverse_edges=_counts(adverse, "label", "relation"),
        drug_disease_conflict_pairs=drug_disease_conflict_pairs,
        indications=_audit_indications(drug_disease),
        identity=_audit_identity(diseases, phenotypes),
        drugs=_audit_drugs(drugs, indications, drug_drug),
        disease_gene_evidence=_numeric_summaries(disease_gene, evidence_fields),
        anatomy_gene_relations=_counts(anatomy_gene, "relation"),
        anatomy_gene_quality=_value_counts(anatomy_gene, _property("call_quality")),
        edge_provenance=_edge_provenance(snapshot),
    )


def validate_audit(report: AuditReport) -> None:
    failures: list[str] = []

    if report.null_node_ids:
        failures.append(f"null node IDs: {report.null_node_ids:,}")

    if report.null_edge_fields:
        failures.append(f"edges with null structural fields: {report.null_edge_fields:,}")

    if report.duplicate_node_ids:
        failures.append(f"duplicate node IDs: {report.duplicate_node_ids:,}")

    if report.orphan_source_edges:
        failures.append(f"edges with missing source nodes: {report.orphan_source_edges:,}")

    if report.orphan_target_edges:
        failures.append(f"edges with missing target nodes: {report.orphan_target_edges:,}")

    if report.directionality_conflicts:
        failures.append(f"relation directionality conflicts: {report.directionality_conflicts:,}")

    if report.largest_connected_component_nodes > report.node_count:
        failures.append("largest connected component has more nodes than the full graph")

    if report.largest_connected_component_edges > report.edge_count:
        failures.append("largest connected component has more edges than the full graph")

    mismatched_views = [path for path, matches in report.typed_view_matches.items() if not matches]

    if mismatched_views:
        failures.append("typed views disagree with unified graph: " + ", ".join(mismatched_views))

    if failures:
        details = "\n".join(f"- {failure}" for failure in failures)
        raise RuntimeError(f"OptimusKG source audit failed:\n{details}")


def write_audit(report: AuditReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(report), indent=2, sort_keys=True) + "\n")
