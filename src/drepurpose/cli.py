__all__ = ("main",)

from argparse import ArgumentParser
from pathlib import Path

from .data.audit import audit_source, validate_audit, write_audit
from .data.benchmark import DEFAULT_BENCHMARK_ROOT, build_benchmark
from .data.source import DEFAULT_SOURCE_ROOT, load_source


def _parser() -> ArgumentParser:
    parser = ArgumentParser(prog="drepurpose")
    commands = parser.add_subparsers(dest="command", required=True)

    data = commands.add_parser("data")
    data_actions = data.add_subparsers(dest="action", required=True)

    audit = data_actions.add_parser("audit")
    audit.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    audit.add_argument(
        "--output",
        type=Path,
        default=Path("data/audit/optimuskg.json"),
    )

    benchmark = commands.add_parser("benchmark")
    benchmark_actions = benchmark.add_subparsers(dest="action", required=True)

    build = benchmark_actions.add_parser("build")
    build.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    build.add_argument("--output", type=Path, default=DEFAULT_BENCHMARK_ROOT)

    return parser


def main() -> None:
    args = _parser().parse_args()

    match args.command, args.action:
        case "data", "audit":
            report = audit_source(load_source(args.source_root))
            write_audit(report, args.output)
            validate_audit(report)

            print(f"dataset: {report.dataset_doi}")
            print(f"nodes: {report.node_count:,}")
            print(f"edges: {report.edge_count:,}")
            print(
                "largest component: "
                f"{report.largest_connected_component_nodes:,} nodes, "
                f"{report.largest_connected_component_edges:,} edges"
            )
            print(f"indications: {report.indications.edges:,}")
            print(f"indication diseases: {report.indications.diseases:,}")
            print(f"indication drugs: {report.indications.drugs:,}")
            print(f"duplicate edge keys: {report.duplicate_edge_keys:,}")
            print(f"drug-disease conflicts: {report.drug_disease_conflict_pairs:,}")
            print(f"endpoint type mismatches: {report.endpoint_type_mismatches:,}")

            for mismatch, count in report.endpoint_type_mismatch_counts.items():
                print(f"  {mismatch}: {count:,}")

            print(
                "disease primary-CUI duplicate groups: "
                f"{report.identity.disease_primary_cui_duplicate_groups:,}"
            )
            print(
                "disease/phenotype shared concept IDs: "
                f"{report.identity.disease_phenotype_shared_concept_ids:,}"
            )
            print(args.output)

        case "benchmark", "build":
            report = build_benchmark(load_source(args.source_root), args.output)

            print(f"nodes: {report.nodes:,}")
            print(f"reasoning edges: {report.edges:,}")
            print(f"candidate drugs: {report.drugs:,}")
            print(f"eligible diseases: {report.diseases:,}")
            print(f"excluded diseases: {report.excluded_diseases:,}")
            print(
                "diseases: "
                f"{report.train_diseases:,} train, "
                f"{report.validation_diseases:,} validation, "
                f"{report.test_diseases:,} test"
            )
            print(
                "indications: "
                f"{report.train_indications:,} train, "
                f"{report.validation_indications:,} validation, "
                f"{report.test_indications:,} test"
            )
            print(report.root)
