from drepurpose.data.sources import (
    ALL_SOURCES,
    PRIMEKG_SOURCES,
    TXGNN_COMMIT,
    TXGNN_SOURCES,
)


def test_source_keys_are_unique() -> None:
    keys = [source.key for source in ALL_SOURCES]

    assert len(keys) == len(set(keys))


def test_source_paths_are_unique() -> None:
    paths = [source.path for source in ALL_SOURCES]

    assert len(paths) == len(set(paths))


def test_primekg_sources_match_txgnn() -> None:
    identifiers = {source.identifier for source in PRIMEKG_SOURCES}

    assert identifiers == {
        "harvard-dataverse:6180626",
        "harvard-dataverse:6180617",
        "harvard-dataverse:6180616",
    }


def test_txgnn_sources_are_commit_pinned() -> None:
    assert len(TXGNN_SOURCES) == 8
    assert len(TXGNN_COMMIT) == 40

    for source in TXGNN_SOURCES:
        assert TXGNN_COMMIT in source.url


def test_all_disease_area_files_are_pinned() -> None:
    disease_sources = [
        source for source in TXGNN_SOURCES if source.key.startswith("txgnn-disease-")
    ]

    assert {source.key for source in disease_sources} == {
        "txgnn-disease-adrenal_gland",
        "txgnn-disease-anemia",
        "txgnn-disease-cardiovascular",
        "txgnn-disease-cell_proliferation",
        "txgnn-disease-mental_health",
    }
