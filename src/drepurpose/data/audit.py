__all__ = (
    "ADVERSE_RELATIONS",
    "THERAPEUTIC_RELATIONS",
    "AuditReport",
    "audit_source",
    "validate_audit",
    "write_audit",
)

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import polars as pl

from .source import SourceSnapshot

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
    parent_edges: int
    parent_drugs: int


@dataclass(frozen=True, slots=True)
class AuditReport:
    dataset_doi: str
    dataset_server: str
    client_version: str
    files: dict[str, dict[str, str | int]]
    node_count: int
    edge_count: int
    node_types: dict[str, int]
    edge_types: dict[str, int]
    edge_relations: dict[str, int]
    null_node_ids: int
    null_edge_fields: int
    duplicate_node_ids: int
    duplicate_edges: int
    orphan_source_edges: int
    orphan_target_edges: int
    endpoint_type_mismatches: int
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


def _scan(path: Path) -> pl.LazyFrame:
    return pl.scan_parquet(path)


def _require_columns(frame: pl.LazyFrame, path: Path, required: tuple[str, ...]) -> None:
    available = set(frame.collect_schema().names())
    missing = set(required) - available

    if missing:
        names = ", ".join(sorted(missing))
        raise RuntimeError(f"{path.name} is missing required columns: {names}")


def _count(frame: pl.LazyFrame) -> int:
    return int(frame.select(pl.len()).collect().item())


def _counts(frame: pl.LazyFrame, *columns: str) -> dict[str, int]:
    result = frame.group_by(*columns).len().sort(*columns).collect()
    counts: dict[str, int] = {}

    for row in result.iter_rows():
        values = row[:-1]
        count = row[-1]
        key = "/".join("null" if value is None else str(value) for value in values)
        counts[key] = int(count)

    return counts


def _value_counts(frame: pl.LazyFrame, expression: pl.Expr) -> dict[str, int]:
    return _counts(frame.select(expression.alias("value")), "value")


def _property(name: str) -> pl.Expr:
    return pl.col("properties").struct.field(name)


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
        histogram={str(count): int(size) for count, size in histogram.iter_rows()},
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


def _nonempty_string(expression: pl.Expr) -> pl.Expr:
    return expression.is_not_null() & (expression != "")


def _audit_identity(
    diseases: pl.LazyFrame,
    phenotypes: pl.LazyFrame,
) -> IdentityAudit:
    disease_primary = diseases.select(
        "id",
        _property("umls_cui").alias("cui"),
    ).filter(_nonempty_string(pl.col("cui")))

    phenotype_primary = phenotypes.select(
        "id",
        _property("umls_cui").alias("cui"),
    ).filter(_nonempty_string(pl.col("cui")))

    duplicate_groups = (
        disease_primary.group_by("cui")
        .agg(pl.col("id").n_unique().alias("nodes"))
        .filter(pl.col("nodes") > 1)
    )

    duplicate_group_count = _count(duplicate_groups)
    duplicate_node_count = int(
        duplicate_groups.select((pl.col("nodes") - 1).sum().fill_null(0)).collect().item()
    )

    shared_primary = _count(
        disease_primary.select("cui")
        .unique()
        .join(
            phenotype_primary.select("cui").unique(),
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

    shared_concepts = _count(
        disease_concepts.join(
            phenotype_concepts,
            on="concept_id",
            how="inner",
        )
    )

    return IdentityAudit(
        disease_primary_cui_duplicate_groups=duplicate_group_count,
        disease_primary_cui_duplicate_nodes=duplicate_node_count,
        disease_phenotype_shared_primary_cuis=shared_primary,
        disease_phenotype_shared_concept_ids=shared_concepts,
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
    parent_drugs = _count(
        pl.concat(
            (
                parent_edges.select(pl.col("from").alias("id")),
                parent_edges.select(pl.col("to").alias("id")),
            )
        ).unique()
    )

    return DrugAudit(
        approval=_value_counts(drug_info, pl.col("is_approved")),
        status=_value_counts(drug_info, pl.col("status")),
        indication_drug_approval=_value_counts(
            indication_drug_info,
            pl.col("is_approved"),
        ),
        parent_edges=_count(parent_edges),
        parent_drugs=parent_drugs,
    )


def _audit_indications(drug_disease: pl.LazyFrame) -> IndicationAudit:
    indications = drug_disease.filter(pl.col("relation") == "INDICATION")
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


def audit_source(snapshot: SourceSnapshot) -> AuditReport:
    nodes = _scan(snapshot.path("nodes.parquet"))
    edges = _scan(snapshot.path("edges.parquet"))
    drugs = _scan(snapshot.path("nodes/drug.parquet"))
    diseases = _scan(snapshot.path("nodes/disease.parquet"))
    phenotypes = _scan(snapshot.path("nodes/phenotype.parquet"))
    drug_disease = _scan(snapshot.path("edges/drug_disease.parquet"))
    drug_phenotype = _scan(snapshot.path("edges/drug_phenotype.parquet"))
    drug_drug = _scan(snapshot.path("edges/drug_drug.parquet"))
    disease_gene = _scan(snapshot.path("edges/disease_gene.parquet"))
    anatomy_gene = _scan(snapshot.path("edges/anatomy_gene.parquet"))

    _require_columns(nodes, snapshot.path("nodes.parquet"), _NODE_COLUMNS)
    _require_columns(edges, snapshot.path("edges.parquet"), _EDGE_COLUMNS)

    node_types = _counts(nodes, "label")
    edge_types = _counts(edges, "label")

    node_labels = nodes.select(
        "id",
        pl.col("label").alias("node_label"),
    )
    typed_edges = (
        edges.select("from", "to", "label")
        .join(
            node_labels,
            left_on="from",
            right_on="id",
            how="left",
        )
        .rename({"node_label": "from_label"})
        .join(
            node_labels,
            left_on="to",
            right_on="id",
            how="left",
        )
        .rename({"node_label": "to_label"})
    )

    label_parts = pl.col("label").str.split_exact("-", 1)
    endpoint_summary = typed_edges.select(
        pl.col("from_label").is_null().sum().alias("orphan_source"),
        pl.col("to_label").is_null().sum().alias("orphan_target"),
        (
            pl.col("from_label").is_not_null()
            & pl.col("to_label").is_not_null()
            & (
                (pl.col("from_label") != label_parts.struct.field("field_0"))
                | (pl.col("to_label") != label_parts.struct.field("field_1"))
            )
        )
        .sum()
        .alias("type_mismatches"),
    ).collect()

    duplicate_node_ids = int(
        nodes.group_by("id")
        .len()
        .filter(pl.col("len") > 1)
        .select((pl.col("len") - 1).sum().fill_null(0))
        .collect()
        .item()
    )

    duplicate_edges = int(
        edges.group_by(
            "from",
            "to",
            "label",
            "relation",
            "undirected",
        )
        .len()
        .filter(pl.col("len") > 1)
        .select((pl.col("len") - 1).sum().fill_null(0))
        .collect()
        .item()
    )

    directionality_conflicts = _count(
        edges.group_by("label", "relation")
        .agg(pl.col("undirected").n_unique().alias("values"))
        .filter(pl.col("values") > 1)
    )

    therapeutic = edges.filter(pl.col("relation").is_in(THERAPEUTIC_RELATIONS))
    adverse = edges.filter(pl.col("relation").is_in(ADVERSE_RELATIONS))

    drug_disease_conflicts = _count(
        drug_disease.filter(pl.col("relation").is_in(THERAPEUTIC_RELATIONS))
        .group_by("from", "to")
        .agg(pl.col("relation").n_unique().alias("relations"))
        .filter(pl.col("relations") > 1)
    )

    indications = drug_disease.filter(pl.col("relation") == "INDICATION")

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

    typed_view_matches = {
        "nodes/drug.parquet": _count(drugs) == node_types.get("DRG", 0),
        "nodes/disease.parquet": _count(diseases) == node_types.get("DIS", 0),
        "nodes/phenotype.parquet": _count(phenotypes) == node_types.get("PHE", 0),
        "edges/drug_disease.parquet": _count(drug_disease) == edge_types.get("DRG-DIS", 0),
        "edges/drug_phenotype.parquet": _count(drug_phenotype) == edge_types.get("DRG-PHE", 0),
        "edges/drug_drug.parquet": _count(drug_drug) == edge_types.get("DRG-DRG", 0),
        "edges/disease_gene.parquet": _count(disease_gene) == edge_types.get("DIS-GEN", 0),
        "edges/anatomy_gene.parquet": _count(anatomy_gene) == edge_types.get("ANA-GEN", 0),
    }

    return AuditReport(
        dataset_doi=snapshot.doi,
        dataset_server=snapshot.server,
        client_version=snapshot.client_version,
        files=files,
        node_count=_count(nodes),
        edge_count=_count(edges),
        node_types=node_types,
        edge_types=edge_types,
        edge_relations=_counts(edges, "label", "relation"),
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
        duplicate_node_ids=duplicate_node_ids,
        duplicate_edges=duplicate_edges,
        orphan_source_edges=int(endpoint_summary["orphan_source"].item()),
        orphan_target_edges=int(endpoint_summary["orphan_target"].item()),
        endpoint_type_mismatches=int(endpoint_summary["type_mismatches"].item()),
        directionality_conflicts=directionality_conflicts,
        self_loops=_count(edges.filter(pl.col("from") == pl.col("to"))),
        typed_view_matches=typed_view_matches,
        therapeutic_edges=_counts(therapeutic, "label", "relation"),
        adverse_edges=_counts(adverse, "label", "relation"),
        drug_disease_conflict_pairs=drug_disease_conflicts,
        indications=_audit_indications(drug_disease),
        identity=_audit_identity(diseases, phenotypes),
        drugs=_audit_drugs(drugs, indications, drug_drug),
        disease_gene_evidence=_numeric_summaries(disease_gene, evidence_fields),
        anatomy_gene_relations=_counts(anatomy_gene, "relation"),
        anatomy_gene_quality=_value_counts(anatomy_gene, _property("call_quality")),
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

    if report.endpoint_type_mismatches:
        failures.append(f"edge endpoint type mismatches: {report.endpoint_type_mismatches:,}")

    if report.directionality_conflicts:
        failures.append(f"relation directionality conflicts: {report.directionality_conflicts:,}")

    mismatched_views = [name for name, matches in report.typed_view_matches.items() if not matches]

    if mismatched_views:
        failures.append("typed views do not match unified tables: " + ", ".join(mismatched_views))

    if report.indications.edges == 0:
        failures.append("no DRG-DIS/INDICATION edges found")

    if failures:
        details = "\n".join(f"- {failure}" for failure in failures)
        raise RuntimeError(f"OptimusKG source audit failed:\n{details}")


def write_audit(report: AuditReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            asdict(report),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
