from hashlib import sha256
from pathlib import Path

from drepurpose.data.fetch import sha256_file


def test_sha256_file(tmp_path: Path) -> None:
    contents = b"drepurpose\n"
    path = tmp_path / "example.txt"
    path.write_bytes(contents)

    assert sha256_file(path) == sha256(contents).hexdigest()
