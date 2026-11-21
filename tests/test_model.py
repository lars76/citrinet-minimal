import glob
import os
import sys
import tarfile
import types

import pytest
import torch
import torch.nn.functional as F
import yaml

from conftest import ROOT
import hubconf
from model import Citrinet, MaskedConv, banded_conv

DATA = os.path.join(ROOT, "tests", "data")
FILES = sorted(glob.glob(os.path.join(DATA, "*.flac")))
REFERENCE = os.path.join(DATA, "nemo_256_ls.pt")


def batch(model):
    audios = [model.load_audio(f) for f in FILES]
    return torch.nn.utils.rnn.pad_sequence(audios, batch_first=True), torch.tensor([len(a) for a in audios])


def items(feat, feat_lens, enc, enc_lens, log_probs):
    return [
        dict(feat=feat[b, :, :n].clone(), enc=enc[b, :, :m].clone(), log_probs=log_probs[b, :m].clone())
        for b, (n, m) in enumerate(zip(feat_lens.tolist(), enc_lens.tolist()))
    ]


def model_items(model):
    audio, lens = batch(model)
    with torch.inference_mode():
        feat, feat_lens = model.features(audio, lens)
        enc, enc_lens = model.encoder(feat, feat_lens)
        log_probs, _ = model(audio, lens)
    return items(feat, feat_lens, enc, enc_lens, log_probs)


def nemo_items(checkpoint):
    from nemo.collections.asr.models import ASRModel

    nemo = ASRModel.restore_from(checkpoint, map_location="cpu").eval()
    nemo.preprocessor.featurizer.pad_to = 0
    audio, lens = batch(Citrinet.from_nemo(checkpoint))
    with torch.inference_mode():
        feat, feat_lens = nemo.preprocessor(input_signal=audio, length=lens)
        enc, enc_lens = nemo.encoder(audio_signal=feat, length=feat_lens)
        return items(feat, feat_lens, enc, enc_lens, nemo.decoder(encoder_output=enc))


def assert_matches(ours, theirs):
    assert len(ours) == len(theirs) == len(FILES)
    for a, b in zip(ours, theirs):
        for key, tolerance in (("feat", 1e-3), ("enc", 1e-2), ("log_probs", 1e-2)):
            assert a[key].shape == b[key].shape
            assert (a[key] - b[key]).abs().max().item() < tolerance
        assert torch.equal(a["log_probs"].argmax(-1), b["log_probs"].argmax(-1))


def test_matches_stored_nemo_reference(model):
    assert_matches(model_items(model), torch.load(REFERENCE, weights_only=True)["items"])


def test_matches_nemo(checkpoint, model):
    pytest.importorskip("nemo")
    assert_matches(model_items(model), nemo_items(checkpoint))


def test_transcribe_matches_librispeech(model):
    refs = dict(line.strip().split(" ", 1) for line in open(os.path.join(DATA, "transcripts.txt")))
    expected = [refs[os.path.basename(f)[:-5]].lower() for f in FILES]
    assert model.transcribe(FILES) == expected
    assert [model.transcribe([f])[0] for f in FILES] == expected


@pytest.mark.parametrize(
    "vocabulary, tokens, text",
    [
        (["boys", "##'", "room"], ["boys", "##'", "room"], "boys' room"),
        (["quilter", "'", "s", "##s"], ["quilter", "'", "s"], "quilter's"),
        (["▁boys'", "▁room"], ["▁boys'", "▁room"], "boys' room"),
        (["▁give", "▁'", "em"], ["▁give", "▁'", "em"], "give 'em"),
        (["钙", "钾", " "], ["钙", " ", "钾"], "钙 钾"),
        (["T", "AH", "M", "[CLS]"], ["T", "AH", "M", "[CLS]"], "T AH M"),
    ],
)
def test_detokenize(vocabulary, tokens, text):
    stub = types.SimpleNamespace(vocabulary=vocabulary)
    assert Citrinet.detokenize(stub, [vocabulary.index(t) for t in tokens]) == text


@pytest.mark.parametrize("stride, padding", [(1, 5), (2, 5), (1, (8, 2)), (3, (6, 4))])
def test_banded_conv_matches_conv(stride, padding):
    torch.manual_seed(0)
    conv = MaskedConv(16, 16, 11, stride, 1, padding, groups=16).conv
    pad = (padding, padding) if isinstance(padding, int) else padding
    x = torch.randn(3, 16, 97)
    expected = F.conv1d(F.pad(x, pad), conv.weight, stride=stride, groups=16)
    torch.testing.assert_close(banded_conv(x, conv, pad), expected, atol=1e-5, rtol=1e-5)


def test_training_ignores_padding_at_batch_size_one(model):
    model.train()
    torch.manual_seed(0)
    x, lens = torch.randn(1, 80, 300), torch.tensor([300])
    outputs = []
    for pad in (torch.zeros(1, 80, 100), torch.randn(1, 80, 100) * 50):
        torch.manual_seed(1)
        y, y_lens = model.encoder(torch.cat([x, pad], -1), lens)
        outputs.append(y[..., : y_lens[0]])
    torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)


def test_transcribe_restores_training_mode(model):
    expected = model.transcribe(FILES)
    model.train()
    running_mean = model.encoder.encoder[0].mconv[2].running_mean.clone()
    assert model.transcribe(FILES) == expected
    assert model.training
    assert torch.equal(running_mean, model.encoder.encoder[0].mconv[2].running_mean)


def test_unsupported_encoder_raises(checkpoint):
    with tarfile.open(checkpoint) as tar:
        member = next(m for m in tar.getmembers() if m.name.endswith("model_config.yaml"))
        cfg = yaml.safe_load(tar.extractfile(member))
    cfg["encoder"]["_target_"] = "nemo.collections.asr.modules.conv_asr.ConvASREncoderAdapter"
    with pytest.raises(ValueError, match="unsupported config: encoder"):
        Citrinet(cfg)


def test_hubconf_rejects_unknown_name():
    with pytest.raises(ValueError, match="choose one of"):
        hubconf.citrinet("stt_en_citrinet_265_ls")


def test_hubconf_loads_local_file(checkpoint):
    assert isinstance(hubconf.citrinet(checkpoint), Citrinet)


if __name__ == "__main__":
    torch.save(dict(torch=str(torch.__version__), items=nemo_items(sys.argv[1])), REFERENCE)
