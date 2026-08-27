__all__ = ("main",)

from argparse import ArgumentParser
from pathlib import Path
from typing import cast

from .data.fetch import audit_sources, fetch_sources
from .data.prepare import prepare_benchmark
from .data.txgnn import DISEASE_AREAS, DiseaseArea


def _parser() -> ArgumentParser:
    parser = ArgumentParser(prog="drepurpose")
    commands = parser.add_subparsers(dest="command", required=True)

    data = commands.add_parser("data")
    data_commands = data.add_subparsers(dest="data_command", required=True)

    fetch = data_commands.add_parser("fetch")
    fetch.add_argument("--root", type=Path, default=Path("data/raw"))
    fetch.add_argument("--force", action="store_true")

    audit = data_commands.add_parser("audit")
    audit.add_argument("--root", type=Path, default=Path("data/raw"))

    prepare = data_commands.add_parser("prepare")
    prepare.add_argument("--area", choices=tuple(DISEASE_AREAS), default="cell_proliferation")
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    prepare.add_argument("--output-root", type=Path, default=Path("data/processed"))
    prepare.add_argument("--force", action="store_true")

    return parser


def main() -> None:
    args = _parser().parse_args()

    match args.command, args.data_command:
        case "data", "fetch":
            fetch_sources(args.root, force=args.force)
        case "data", "audit":
            audit_sources(args.root)
        case "data", "prepare":
            path = prepare_benchmark(
                args.raw_root,
                args.output_root,
                cast(DiseaseArea, args.area),
                seed=args.seed,
                force=args.force,
            )
            print(path)
