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

    return cast(SourceManifest, json.loads(path.read_text()))


def _download(source: SourceFile, path: Path) -> tuple[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_suffix(path.suffix + ".part")
    temporary.unlink(missing_ok=True)

    request = Request(source.url, headers={"User-Agent": "drepurpose/0.1"})
    digest = sha256()
    size = 0

    try:
        with urlopen(request) as response, temporary.open("wb") as file:
            content_length = response.headers.get("Content-Length")
            total = int(content_length) if content_length is not None else None

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

        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    return digest.hexdigest(), size


def fetch_sources(root: Path, *, force: bool = False) -> None:
    root.mkdir(parents=True, exist_ok=True)

    manifest_path = root / "manifest.json"
    previous = _load_manifest(manifest_path)

    if previous is not None and previous["txgnn_commit"] != TXGNN_COMMIT:
        raise RuntimeError(f"TxGNN commit mismatch: {previous['txgnn_commit']} != {TXGNN_COMMIT}")

    previous_files = previous["files"] if previous is not None else {}
    records: dict[str, SourceRecord] = {}

    for source in ALL_SOURCES:
        path = root / source.path

        if path.exists() and not force:
            digest = sha256_file(path)
            size = path.stat().st_size

            previous_record = previous_files.get(source.key)
            if previous_record is not None and previous_record["sha256"] != digest:
                raise RuntimeError(f"SHA-256 mismatch: {path}")
        else:
            digest, size = _download(source, path)

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

    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(manifest_path)


def _csv_columns(path: Path) -> tuple[str, ...]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return tuple(next(csv.reader(file)))


def audit_sources(root: Path) -> None:
    manifest_path = root / "manifest.json"
    manifest = _load_manifest(manifest_path)

    if manifest is None:
        raise FileNotFoundError(manifest_path)

    if manifest["txgnn_commit"] != TXGNN_COMMIT:
        raise RuntimeError(f"TxGNN commit mismatch: {manifest['txgnn_commit']} != {TXGNN_COMMIT}")

    for source in ALL_SOURCES:
        path = root / source.path

        if not path.exists():
            raise FileNotFoundError(path)

        record = manifest["files"].get(source.key)
        if record is None:
            raise RuntimeError(f"Missing manifest entry: {source.key}")

        digest = sha256_file(path)
        if digest != record["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch: {path}")

        size_mib = path.stat().st_size / 1024**2
        print(f"{source.key}  {size_mib:.1f} MiB  {digest[:12]}")

        if path.suffix == ".csv":
            print("  " + ", ".join(_csv_columns(path)))
