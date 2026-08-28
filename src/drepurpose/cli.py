__all__ = ("main",)

from argparse import ArgumentParser
from pathlib import Path

from .data.audit import audit_source, validate_audit, write_audit
from .data.source import fetch_source


def _parser() -> ArgumentParser:
    parser = ArgumentParser(prog="drepurpose")
    commands = parser.add_subparsers(
        dest="command",
        required=True,
    )

    data = commands.add_parser("data")
    data_commands = data.add_subparsers(
        dest="data_command",
        required=True,
    )

    fetch = data_commands.add_parser("fetch")
    fetch.add_argument("--force", action="store_true")

    audit = data_commands.add_parser("audit")
    audit.add_argument("--force", action="store_true")
    audit.add_argument(
        "--output",
        type=Path,
        default=Path("data/audit/optimuskg.json"),
    )

    return parser


def main() -> None:
    args = _parser().parse_args()

    match args.command, args.data_command:
        case "data", "fetch":
            snapshot = fetch_source(force=args.force)

            print(f"dataset: {snapshot.doi}")
            print(f"client: {snapshot.client_version}")

            for file in snapshot.files:
                print(f"{file.relative_path}: {file.local_path}")

        case "data", "audit":
            report = audit_source(fetch_source(force=args.force))
            validate_audit(report)
            write_audit(report, args.output)

            print(f"nodes: {report.node_count:,}")
            print(f"edges: {report.edge_count:,}")
            print(f"indications: {report.indications.edges:,}")
            print(f"indication diseases: {report.indications.diseases:,}")
            print(f"indication drugs: {report.indications.drugs:,}")
            print(f"duplicate edges: {report.duplicate_edges:,}")
            print(f"drug-disease conflict pairs: {report.drug_disease_conflict_pairs:,}")
            print(
                "disease primary-CUI duplicate groups: "
                f"{report.identity.disease_primary_cui_duplicate_groups:,}"
            )
            print(
                "disease/phenotype shared concept IDs: "
                f"{report.identity.disease_phenotype_shared_concept_ids:,}"
            )
            print(args.output)
