import os

import torch

from model import Citrinet

dependencies = ["torch", "yaml", "soundfile"]
RELEASE = "https://github.com/lars76/citrinet-minimal/releases/download/v1.0/"
NGC = "https://api.ngc.nvidia.com/v2/models/nvidia/nemo/"
NGC_TERMS = "https://developer.nvidia.com/legal/terms"
URLS = {
    name: f"{RELEASE}{name}.nemo"
    for name in (
        "stt_en_citrinet_256_ls",
        "stt_en_citrinet_384_ls",
        "stt_en_citrinet_512_ls",
        "stt_en_citrinet_768_ls",
        "stt_en_citrinet_1024_gamma_0_25",
        "stt_zh_citrinet_1024_gamma_0_25",
        "stt_uk_citrinet_1024_gamma_0_25",
    )
} | {
    name: f"{NGC}{name}/versions/{version}/files/{name}.nemo"
    for name, version in (
        ("stt_en_citrinet_256", "1.0.0rc1"),
        ("stt_en_citrinet_512", "1.0.0rc1"),
        ("stt_de_citrinet_1024", "1.5.0"),
        ("stt_es_citrinet_512", "1.0.0"),
        ("stt_es_citrinet_1024_gamma_0_25", "1.8.0"),
        ("stt_fr_citrinet_1024_gamma_0_25", "1.5"),
    )
}


def citrinet(name="stt_en_citrinet_256_ls"):
    if os.path.isfile(name):
        return Citrinet.from_nemo(name)
    if name not in URLS:
        raise ValueError(f"unknown checkpoint {name!r}, choose one of: {', '.join(URLS)}")
    path = os.path.join(torch.hub.get_dir(), "checkpoints", f"{name}.nemo")
    if not os.path.exists(path):
        if URLS[name].startswith(NGC):
            print(f"Downloading {name} from NVIDIA NGC. It is under the NVIDIA Technology Access Terms of Use: {NGC_TERMS}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.hub.download_url_to_file(URLS[name], path)
    return Citrinet.from_nemo(path)
