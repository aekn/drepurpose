__all__ = ("main",)

from argparse import ArgumentParser
from pathlib import Path

from drepurpose.data.fetch import audit_sources, fetch_sources


def _parser() -> ArgumentParser:
    parser = ArgumentParser(prog="drepurpose")
    commands = parser.add_subparsers(dest="command", required=True)

    data = commands.add_parser("data", help="Manage benchmark data")
    data_commands = data.add_subparsers(dest="data_command", required=True)

    fetch = data_commands.add_parser("fetch", help="Fetch pinned benchmark sources")
    fetch.add_argument("--root", type=Path, default=Path("data/raw"))
    fetch.add_argument("--force", action="store_true")

    audit = data_commands.add_parser(
        "audit", help="Verify downloaded benchmark sources"
    )
    audit.add_argument("--root", type=Path, default=Path("data/raw"))

    return parser


def main() -> None:
    args = _parser().parse_args()

    match args.command, args.data_command:
        case "data", "fetch":
            fetch_sources(args.root, force=args.force)
        case "data", "audit":
            audit_sources(args.root)
        case _:
            pass
