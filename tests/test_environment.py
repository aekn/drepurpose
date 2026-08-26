import sys
from importlib.metadata import version

import torch


def test_python_version() -> None:
    assert sys.version_info[:2] == (3, 14)


def test_framework_versions() -> None:
    assert torch.__version__.split("+")[0] == "2.12.1"
    assert version("torch-geometric") == "2.8.0"


def test_tensor_operations() -> None:
    values = torch.tensor([1.0, 2.0, 3.0])

    assert values.sum().item() == 6.0
