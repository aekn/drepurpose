__all__ = (
    "ALL_SOURCES",
    "PRIMEKG_SOURCES",
    "TXGNN_COMMIT",
    "TXGNN_SOURCES",
    "SourceFile",
)

from dataclasses import dataclass

TXGNN_COMMIT = "1000aac7120e0022af09a3ef93dba219e16a094b"

_DATAVERSE = "https://dataverse.harvard.edu/api/access/datafile"
_TXGNN_RAW = f"https://raw.githubusercontent.com/mims-harvard/TxGNN/{TXGNN_COMMIT}"


@dataclass(frozen=True, slots=True)
class SourceFile:
    key: str
    path: str
    url: str
    identifier: str


PRIMEKG_SOURCES = (
    SourceFile(
        key="primekg-kg",
        path="primekg/kg.csv",
        url=f"{_DATAVERSE}/6180626",
        identifier="harvard-dataverse:6180626",
    ),
    SourceFile(
        key="primekg-nodes",
        path="primekg/nodes.csv",
        url=f"{_DATAVERSE}/6180617",
        identifier="harvard-dataverse:6180617",
    ),
    SourceFile(
        key="primekg-edges",
        path="primekg/edges.csv",
        url=f"{_DATAVERSE}/6180616",
        identifier="harvard-dataverse:6180616",
    ),
)

TXGNN_SOURCES = (
    SourceFile(
        key="txgnn-human-do",
        path="txgnn-1000aac/HumanDO.obo",
        url=f"{_TXGNN_RAW}/txgnn/data_splits/HumanDO.obo",
        identifier=f"github:mims-harvard/TxGNN@{TXGNN_COMMIT}:HumanDO.obo",
    ),
    SourceFile(
        key="txgnn-mondo-references",
        path="txgnn-1000aac/mondo_references.csv",
        url=f"{_TXGNN_RAW}/txgnn/data_splits/mondo_references.csv",
        identifier=f"github:mims-harvard/TxGNN@{TXGNN_COMMIT}:mondo_references.csv",
    ),
    SourceFile(
        key="txgnn-grouped-diseases",
        path="txgnn-1000aac/kg_grouped_diseases_bert_map.csv",
        url=f"{_TXGNN_RAW}/txgnn/data_splits/kg_grouped_diseases_bert_map.csv",
        identifier=(
            f"github:mims-harvard/TxGNN@{TXGNN_COMMIT}:kg_grouped_diseases_bert_map.csv"
        ),
    ),
)

ALL_SOURCES = PRIMEKG_SOURCES + TXGNN_SOURCES
