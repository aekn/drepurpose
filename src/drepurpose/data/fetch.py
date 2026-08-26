__all__ = ("audit_sources", "fetch_sources", "sha256_file")

import csv
import json
from hashlib import sha256
from pathlib import Path
from typing import TypedDict, cast
from urllib.request import Request, urlopen

from tqdm import tqdm

from .sources import ALL_SOURCES, TXGNN_COMMIT, SourceFile

_CHUNK_SIZE = 1024 * 1024


class SourceRecord(TypedDict):
    identifier: str
    path: str
    sha256: str
    size: int
    url: str


class SourceManifest(TypedDict):
    dataset: str
    benchmark: str
    txgnn_commit: str
    files: dict[str, SourceRecord]


def sha256_file(path: Path) -> str:
    digest = sha256()

    with path.open("rb") as file:
        while chunk := file.read(_CHUNK_SIZE):
            digest.update(chunk)

    return digest.hexdigest()


def _load_manifest(path: Path) -> SourceManifest | None:
    if not path.exists():
        return None

    return cast(SourceManifest, json.loads(path.read_text(encoding="utf-8")))


def _download(source: SourceFile, destination: Path) -> tuple[str, int]:
    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary = destination.with_name(f"{destination.name}.part")
    temporary.unlink(missing_ok=True)

    request = Request(source.url, headers={"User-Agent": "drepurpose/0.1"})

    digest = sha256()
    size = 0

    try:
        with urlopen(request) as response, temporary.open("wb") as file:
            content_length = response.headers.get("Content-Length")
            total = int(content_length) if content_length else None

            with tqdm(
                total=total,
                desc=source.key,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as progress:
                while chunk := response.read(_CHUNK_SIZE):
                    file.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
                    progress.update(len(chunk))

        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    return digest.hexdigest(), size


def fetch_sources(root: Path, *, force: bool = False) -> None:
    root.mkdir(parents=True, exist_ok=True)

    manifest_path = root / "manifest.json"
    previous = _load_manifest(manifest_path)

    if previous is not None and previous["txgnn_commit"] != TXGNN_COMMIT:
        raise RuntimeError(
            f"{manifest_path} was created from TxGNN commit "
            f"{previous['txgnn_commit']}, expected {TXGNN_COMMIT}."
        )

    previous_files = previous["files"] if previous is not None else {}
    records: dict[str, SourceRecord] = {}

    for source in ALL_SOURCES:
        destination = root / source.path

        if destination.exists() and not force:
            digest = sha256_file(destination)
            size = destination.stat().st_size

            previous_record = previous_files.get(source.key)
            if previous_record is not None and previous_record["sha256"] != digest:
                raise RuntimeError(
                    f"{destination} does not match its recorded SHA-256. "
                    "Delete it or fetch again with --force."
                )

            print(f"Found {source.path}")
        else:
            digest, size = _download(source, destination)

        records[source.key] = {
            "identifier": source.identifier,
            "path": source.path,
            "sha256": digest,
            "size": size,
            "url": source.url,
        }

    manifest: SourceManifest = {
        "dataset": "PrimeKG",
        "benchmark": "TxGNN/BioPathNet zero-shot disease-area",
        "txgnn_commit": TXGNN_COMMIT,
        "files": records,
    }

    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {manifest_path}")


def _csv_columns(path: Path) -> tuple[str, ...]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return tuple(next(csv.reader(file)))


def audit_sources(root: Path) -> None:
    manifest_path = root / "manifest.json"
    manifest = _load_manifest(manifest_path)

    if manifest is None:
        raise FileNotFoundError(
            f"No manifest found at {manifest_path}. Fetch the data first."
        )

    if manifest["txgnn_commit"] != TXGNN_COMMIT:
        raise RuntimeError(
            f"Manifest uses TxGNN commit {manifest['txgnn_commit']}, "
            f"expected {TXGNN_COMMIT}."
        )

    for source in ALL_SOURCES:
        path = root / source.path

        if not path.exists():
            raise FileNotFoundError(path)

        expected = manifest["files"].get(source.key)
        if expected is None:
            raise RuntimeError(f"{source.key} is missing from {manifest_path}")

        digest = sha256_file(path)

        if digest != expected["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch for {path}")

        size_mib = path.stat().st_size / 1024**2
        print(f"{source.key}: {size_mib:.1f} MiB  sha256={digest[:12]}...")

        if path.suffix == ".csv":
            print(f"  columns: {', '.join(_csv_columns(path))}")

    print("All source files match the manifest.")
