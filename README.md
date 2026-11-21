# citrinet-minimal

A minimal, single-file PyTorch implementation of NVIDIA's Citrinet speech recognition model. Citrinet is an extremely fast architecture because it is based on depthwise-separable 1D convolutions and does not use attention layers. It loads the original NeMo `.nemo` checkpoints and produces the same outputs as NeMo without needing NeMo.

## Quick start

Tested with Python 3.13 and PyTorch 2.15.

```python
import torch

model = torch.hub.load("lars76/citrinet-minimal", "citrinet", name="stt_en_citrinet_256_ls", trust_repo=True)
print(model.transcribe(["audio.wav"]))
```

Run the script with `uv run --with torch --with soundfile --with pyyaml --with soxr python script.py`. `name` is one of the checkpoints below, or a path to a local `.nemo` file.

`transcribe` reads any format that soundfile supports, mixes to mono, resamples to 16 kHz, sorts the files by length and runs them in batches. It runs on CPU, MPS or CUDA, wherever the model is.

The tests in `tests/` check that `model.py` matches NeMo 2.4 at every stage on three LibriSpeech files. On every push they compare against a stored NeMo reference on Linux and macOS. On Linux x86 the outputs were identical bit for bit, and on macOS they differed only by floating point rounding, with no change in any predicted token. The manual CI workflow `nemo` installs NeMo and compares against it directly. Run the tests with `uv run --with torch --with soundfile --with pyyaml --with soxr --with pytest pytest`. In an environment with NeMo, `python tests/test_model.py CHECKPOINT.nemo` rewrites the stored reference.

## Checkpoints

All checkpoints are NVIDIA's original files. "Release" checkpoints are CC-BY-4.0 and attached to the GitHub release. "NGC" checkpoints are under the [NVIDIA Technology Access Terms of Use](https://developer.nvidia.com/legal/terms), which forbid redistribution and allow use for yourself only, without modification or fine-tuning. For those, `torch.hub.load` downloads the file directly from NVIDIA, without a login, and prints a notice about the terms.

The error rates are from NVIDIA's model cards, in percent, with greedy decoding. WER is the word error rate and CER the character error rate. MLS is Multilingual LibriSpeech and MCV is Mozilla Common Voice.

| Checkpoint | Source | Language | Params | Size | Training data | Error rate (test set) |
|---|---|---|---|---|---|---|
| `stt_en_citrinet_256_ls` | Release | English | 9.8M | 35 MB | LibriSpeech 960 h | WER 3.78 / 9.6 (LibriSpeech clean / other) |
| `stt_en_citrinet_384_ls` | Release | English | 21.0M | 75 MB | LibriSpeech 960 h | WER 3.2 / 7.9 (LibriSpeech clean / other) |
| `stt_en_citrinet_512_ls` | Release | English | 36.5M | 130 MB | LibriSpeech 960 h | WER 3.1 / 7.8 (LibriSpeech clean / other) |
| `stt_en_citrinet_768_ls` | Release | English | 81.0M | 289 MB | LibriSpeech 960 h | WER 2.6 / 6.4 (LibriSpeech clean / other) |
| `stt_en_citrinet_1024_gamma_0_25` | Release | English | 141.2M | 461 MB | NVIDIA ASR Set, 7,000+ h | WER 3.4 / 7.6 (LibriSpeech clean / other) |
| `stt_en_citrinet_256` | NGC | English | 10.3M | 37 MB | NVIDIA ASR Set, 7,000+ h | WER 4.4 / 10.7 (LibriSpeech clean / other) |
| `stt_en_citrinet_512` | NGC | English | 37.0M | 133 MB | NVIDIA ASR Set, 7,000+ h | WER 3.7 / 8.9 (LibriSpeech clean / other) |
| `stt_de_citrinet_1024` | NGC | German | 141.2M | 508 MB | 2,300 h, pretrained on 20,000 h of unlabeled speech | WER 5.07 (MLS test), 7.59 (MCV 7.0 test) |
| `stt_es_citrinet_512` | NGC | Spanish | 37.0M | 134 MB | about 1,065 h, English encoder as start | WER 5.2 (MLS test), 10.3 (MCV test) |
| `stt_es_citrinet_1024_gamma_0_25` | NGC | Spanish | 141.2M | 453 MB | 1,340 h, English 1024 model as start | WER 4.1 (MLS test), 6.8 (MCV 7.0 test) |
| `stt_fr_citrinet_1024_gamma_0_25` | NGC | French | 141.2M | 462 MB | 1,500+ h | WER 5.53 (MLS test), 12.2 (MCV 7.0 test) |
| `stt_zh_citrinet_1024_gamma_0_25` | Release | Mandarin | 145.5M | 530 MB | AISHELL-2 | CER 5.1 (AISHELL-2 iOS test) |
| `stt_uk_citrinet_1024_gamma_0_25` | Release | Ukrainian | 141.2M | 540 MB | Common Voice 10 | WER 5.02 (Common Voice 10 test) |

We checked the English release numbers with `model.py` on the full LibriSpeech test sets, using greedy decoding and no language model. Our WER matched every card to the precision the card gives.

## License

The code is MIT licensed, see `LICENSE`.

The checkpoints in the release are NVIDIA's, licensed under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) and redistributed unchanged. Original sources: [stt_en_citrinet_256_ls](https://huggingface.co/nvidia/stt_en_citrinet_256_ls), [stt_en_citrinet_384_ls](https://huggingface.co/nvidia/stt_en_citrinet_384_ls), [stt_en_citrinet_512_ls](https://huggingface.co/nvidia/stt_en_citrinet_512_ls), [stt_en_citrinet_768_ls](https://huggingface.co/nvidia/stt_en_citrinet_768_ls), [stt_en_citrinet_1024_gamma_0_25](https://huggingface.co/nvidia/stt_en_citrinet_1024_gamma_0_25), [stt_zh_citrinet_1024_gamma_0_25](https://huggingface.co/nvidia/stt_zh_citrinet_1024_gamma_0_25), [stt_uk_citrinet_1024_gamma_0_25](https://huggingface.co/nvidia/stt_uk_citrinet_1024_gamma_0_25).

The three audio files in `tests/data` are from LibriSpeech test-clean, licensed under [CC-BY-4.0](https://www.openslr.org/12/).

The NGC checkpoints are not part of this repository. `hubconf.py` only points to NVIDIA's download URLs. They are under the [NVIDIA Technology Access Terms of Use](https://developer.nvidia.com/legal/terms), which allow use for yourself only and forbid redistribution and modification.

Citrinet is described in: Somshubra Majumdar, Jagadeesh Balam, Oleksii Hrinchuk, Vitaly Lavrukhin, Vahid Noroozi, Boris Ginsburg. "Citrinet: Closing the Gap between Non-Autoregressive and Autoregressive End-to-End Models for Automatic Speech Recognition." [arXiv:2104.01721](https://arxiv.org/abs/2104.01721), 2021.
