__all__ = (
    "DATASET_DOI",
    "DEFAULT_SOURCE_ROOT",
    "EDGE_FILES",
    "LCC_FILES",
    "NODE_FILES",
    "SOURCE_FILES",
    "SourceFile",
    "SourceSnapshot",
    "load_source",
)


from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

DATASET_DOI = "doi:10.7910/DVN/IYNGEV"
DEFAULT_SOURCE_ROOT = Path("data/source/optimuskg")

LCC_FILES = (
    "largest_connected_component_nodes.parquet",
    "largest_connected_component_edges.parquet",
)

NODE_FILES = {
    "GEN": "nodes/gene.parquet",
    "DIS": "nodes/disease.parquet",
    "BPO": "nodes/biological_process.parquet",
    "PHE": "nodes/phenotype.parquet",
    "DRG": "nodes/drug.parquet",
    "ANA": "nodes/anatomy.parquet",
    "MFN": "nodes/molecular_function.parquet",
    "CCO": "nodes/cellular_component.parquet",
    "PWY": "nodes/pathway.parquet",
    "EXP": "nodes/exposure.parquet",
}

EDGE_FILES = {
    "DIS-GEN": "edges/disease_gene.parquet",
    "ANA-GEN": "edges/anatomy_gene.parquet",
    "DRG-DRG": "edges/drug_drug.parquet",
    "PHE-GEN": "edges/phenotype_gene.parquet",
    "GEN-GEN": "edges/gene_gene.parquet",
    "DIS-PHE": "edges/disease_phenotype.parquet",
    "BPO-GEN": "edges/biological_process_gene.parquet",
    "DRG-DIS": "edges/drug_disease.parquet",
    "MFN-GEN": "edges/molecular_function_gene.parquet",
    "DRG-PHE": "edges/drug_phenotype.parquet",
    "PWY-GEN": "edges/pathway_gene.parquet",
    "BPO-BPO": "edges/biological_process_biological_process.parquet",
    "DIS-DIS": "edges/disease_disease.parquet",
    "CCO-GEN": "edges/cellular_component_gene.parquet",
    "DRG-GEN": "edges/drug_gene.parquet",
    "PHE-PHE": "edges/phenotype_phenotype.parquet",
    "MFN-MFN": "edges/molecular_function_molecular_function.parquet",
    "PWY-PWY": "edges/pathway_pathway.parquet",
    "EXP-GEN": "edges/exposure_gene.parquet",
    "EXP-DIS": "edges/exposure_disease.parquet",
    "EXP-EXP": "edges/exposure_exposure.parquet",
    "EXP-BPO": "edges/exposure_biological_process.parquet",
    "ANA-ANA": "edges/anatomy_anatomy.parquet",
    "CCO-CCO": "edges/cellular_component_cellular_component.parquet",
    "EXP-MFN": "edges/exposure_molecular_function.parquet",
    "EXP-CCO": "edges/exposure_cellular_component.parquet",
    "DRG-BPO": "edges/drug_biological_process.parquet",
}

SOURCE_FILES = (
    "nodes.parquet",
    "edges.parquet",
    *LCC_FILES,
    *NODE_FILES.values(),
    *EDGE_FILES.values(),
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
    root: Path
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


def load_source(root: Path = DEFAULT_SOURCE_ROOT) -> SourceSnapshot:
    root = root.expanduser().resolve()
    missing = [path for path in SOURCE_FILES if not (root / path).is_file()]

    if missing:
        shown = "\n".join(f"- {path}" for path in missing[:10])
        remaining = len(missing) - 10
        suffix = f"\n- ... and {remaining} more" if remaining else ""

        raise RuntimeError(f"Incomplete OptimusKG source at {root}:\n{shown}{suffix}")

    files = tuple(
        SourceFile(
            relative_path=path,
            local_path=root / path,
            size=(root / path).stat().st_size,
            sha256=_sha256_file(root / path),
        )
        for path in SOURCE_FILES
    )

    return SourceSnapshot(
        doi=DATASET_DOI,
        root=root,
        files=files,
    )
