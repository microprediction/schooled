import torch
import math


def test_mps_available():
    # this ensures that the current MacOS version is at least 12.3+
    assert torch.backends.mps.is_available()


def test_mps_built():
    # this ensures that the current PyTorch installation was built with MPS activated.
    assert torch.backends.mps.is_built()


if __name__=='__main__':
    test_mps_available()
    test_mps_built()

    print()