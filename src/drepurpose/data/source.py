from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

import optimuskg

DATASET_DOI = "doi:10.7910/DVN/IYNGEV"

AUDIT_FILES = (
    "nodes.parquet",
    "edges.parquet",
    "nodes/drug.parquet",
    "nodes/disease.parquet",
    "nodes/phenotype.parquet",
    "edges/drug_disease.parquet",
    "edges/drug_phenotype.parquet",
    "edges/drug_drug.parquet",
    "edges/disease_gene.parquet",
    "edges/anatomy_gene.parquet",
)

_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True, slots=True)
class SourceFile:
    relative_path: str
    local_path: Path
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class SourceSnapshot:
    doi: str
    server: str
    client_version: str
    files: tuple[SourceFile, ...]

    def path(self, relative_path: str) -> Path:
        for file in self.files:
            if file.relative_path == relative_path:
                return file.local_path

        raise KeyError(relative_path)


def _sha256_file(path: Path) -> str:
    digest = sha256()

    with path.open("rb") as file:
        while chunk := file.read(_CHUNK_SIZE):
            digest.update(chunk)

    return digest.hexdigest()


def fetch_source(*, force: bool = False) -> SourceSnapshot:
    optimuskg.set_doi(DATASET_DOI)

    files = []

    for relative_path in AUDIT_FILES:
        local_path = Path(optimuskg.get_file(relative_path, force=force))
        files.append(
            SourceFile(
                relative_path=relative_path,
                local_path=local_path,
                size=local_path.stat().st_size,
                sha256=_sha256_file(local_path),
            )
        )

    return SourceSnapshot(
        doi=optimuskg.get_doi(),
        server=optimuskg.get_server(),
        client_version=version("optimuskg"),
        files=tuple(files),
    )
