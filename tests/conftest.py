import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CHECKPOINT_URL = "https://huggingface.co/nvidia/stt_en_citrinet_256_ls/resolve/main/stt_en_citrinet_256_ls.nemo"


@pytest.fixture(scope="session")
def checkpoint():
    path = os.path.join(torch.hub.get_dir(), "checkpoints", "stt_en_citrinet_256_ls.nemo")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.hub.download_url_to_file(CHECKPOINT_URL, path)
    return path


@pytest.fixture
def model(checkpoint):
    from model import Citrinet

    return Citrinet.from_nemo(checkpoint)
