import io
import math
import os
import re
import sys
import tarfile

import torch
import torch.nn.functional as F
import yaml
from torch import nn

ACTIVATIONS = {
    "relu": nn.ReLU,
    "hardtanh": lambda: nn.Hardtanh(0.0, 20.0),
    "selu": nn.SELU,
    "swish": nn.SiLU,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
}
SPECIAL_TOKEN = re.compile(r"\[[A-Z]+\]|<[a-z/]+>")


def time_mask(lens, length):
    return torch.arange(length, device=lens.device)[None] < lens[:, None]


def banded_conv(x, conv, padding, block=32):
    w, k, s = conv.weight[:, 0], conv.kernel_size[0], conv.stride[0]
    span = s * (block - 1) + k
    band = F.pad(w, (0, span + s - k)).repeat(1, block)[:, : block * span].view(-1, block, span)
    length = x.shape[-1] + sum(padding)
    n = -(-((length - k) // s + 1) // block)
    x = F.pad(x, (padding[0], padding[1] + s * block * (n - 1) + span - length))
    y = torch.einsum("bcns,cls->bcnl", x.unfold(-1, span, s * block), band).flatten(2)
    return y[..., : (length - k) // s + 1]


def require(condition, message):
    if not condition:
        raise ValueError(f"unsupported config: {message}")


class Features(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        require(cfg.get("frame_splicing", 1) == 1, "frame_splicing")
        require(cfg.get("normalize", "per_feature") in ("per_feature", "all_features", None, False), "normalize")
        self.sample_rate = cfg.get("sample_rate", 16000)
        self.win_length = int(cfg.get("window_size", 0.02) * self.sample_rate)
        self.hop_length = int(cfg.get("window_stride", 0.01) * self.sample_rate)
        self.n_fft = cfg.get("n_fft") or 2 ** math.ceil(math.log2(self.win_length))
        self.exact_pad = cfg.get("exact_pad", False)
        self.preemph = cfg.get("preemph", 0.97)
        self.mag_power = cfg.get("mag_power", 2.0)
        self.log = cfg.get("log", True)
        self.guard_type = cfg.get("log_zero_guard_type", "add")
        self.guard = cfg.get("log_zero_guard_value", 2**-24)
        self.normalize = cfg.get("normalize", "per_feature")
        n_bins = self.n_fft // 2 + 1
        self.register_buffer("window", torch.zeros(self.win_length))
        self.register_buffer("fb", torch.zeros(1, cfg.get("features", 64), n_bins))

    def forward(self, x, lens):
        pad = (self.n_fft - self.hop_length) // 2 if self.exact_pad else self.n_fft // 2
        out_lens = torch.floor_divide(lens + 2 * pad - self.n_fft, self.hop_length) + 1
        out_lens = torch.where(lens == 0, 0, out_lens).long()
        if self.exact_pad:
            x = F.pad(x[:, None], (pad, pad), "reflect")[:, 0]
        if self.preemph is not None:
            x = torch.cat((x[:, :1], x[:, 1:] - self.preemph * x[:, :-1]), dim=1)
        x = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=not self.exact_pad,
            window=self.window.float(),
            return_complex=True,
        )
        x = torch.sqrt(x.real.pow(2) + x.imag.pow(2))
        if self.mag_power != 1.0:
            x = x.pow(self.mag_power)
        x = torch.matmul(self.fb.to(x.dtype), x)
        if self.log:
            guard = {"tiny": torch.finfo(x.dtype).tiny, "eps": torch.finfo(x.dtype).eps}.get(self.guard, self.guard)
            x = torch.log(x + guard) if self.guard_type == "add" else torch.log(torch.clamp(x, min=guard))
        valid = time_mask(out_lens, x.shape[-1])
        if self.normalize == "per_feature":
            count = valid.sum(1)[:, None]
            mean = torch.where(valid[:, None], x, 0.0).sum(2) / count
            std = torch.sqrt(torch.where(valid[:, None], x - mean[..., None], 0.0).pow(2).sum(2) / (count - 1.0))
            std = std.masked_fill(std.isnan(), 0.0) + 1e-5
            x = (x - mean[..., None]) / std[..., None]
        elif self.normalize == "all_features":
            stats = [(x[i, :, :n].mean(), x[i, :, :n].std()) for i, n in enumerate(out_lens.tolist())]
            mean = torch.stack([m for m, _ in stats])[:, None, None]
            std = torch.stack([s for _, s in stats])[:, None, None] + 1e-5
            x = (x - mean) / std
        return x.masked_fill(~valid[:, None], 0.0), out_lens


class MaskedConv(nn.Module):
    def __init__(self, cin, cout, kernel, stride=1, dilation=1, padding=0, groups=1):
        super().__init__()
        self.padding = padding
        symmetric = isinstance(padding, int)
        self.conv = nn.Conv1d(cin, cout, kernel, stride, padding if symmetric else 0, dilation, groups, bias=False)
        total_pad = 2 * padding if symmetric else sum(padding)
        self.keeps_length = stride == 1 and total_pad == dilation * (kernel - 1)
        self.length_offset = total_pad - dilation * (kernel - 1) - 1

    def forward(self, x, lens):
        if self.training or (x.shape[0] > 1 and self.conv.kernel_size[0] > 1):
            x = x * time_mask(lens, x.shape[-1])[:, None].to(x.dtype)
        if not self.keeps_length:
            lens = torch.div(lens + self.length_offset, self.conv.stride[0], rounding_mode="trunc") + 1
        padding = (self.padding, self.padding) if isinstance(self.padding, int) else self.padding
        if x.device.type == "cpu" and not torch.backends.mkldnn.is_available() and self.conv.groups > 1 and self.conv.dilation[0] == 1:
            return banded_conv(x, self.conv, padding), lens
        if not isinstance(self.padding, int):
            x = F.pad(x, self.padding)
        return self.conv(x), lens


class SqueezeExcite(nn.Module):
    def __init__(self, channels, ratio, activation):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // ratio, bias=False),
            activation,
            nn.Linear(channels // ratio, channels, bias=False),
        )

    def forward(self, x, lens):
        mask = time_mask(lens, x.shape[-1])[:, None]
        x = x.masked_fill(~mask, 0.0)
        y = x.sum(-1, keepdim=True) / mask.sum(-1, keepdim=True).to(x.dtype)
        y = torch.sigmoid(self.fc(y.transpose(1, -1)).transpose(1, -1))
        return x * y, lens


def same_padding(kernel, dilation, future_context):
    if future_context < 0 or kernel <= future_context:
        return dilation * (kernel - 1) // 2
    if dilation > 1:
        return (dilation * kernel - 1 - dilation * future_context, dilation * future_context)
    return (kernel - 1 - future_context, future_context)


class Block(nn.Module):
    def __init__(self, cin, lcfg, activation, residual_mode):
        super().__init__()
        require(not lcfg.get("residual_dense", False), "residual_dense")
        require(lcfg.get("groups", 1) == 1, "groups")
        require(lcfg.get("heads", -1) == -1, "heads")
        residual_mode = lcfg.get("residual_mode", residual_mode)
        require(residual_mode in ("add", "stride_add"), f"residual_mode {residual_mode}")
        channels, repeat = lcfg["filters"], lcfg["repeat"]
        stride, dilation = lcfg["stride"][0], lcfg["dilation"][0]
        kernel = max(int(lcfg["kernel"][0] * float(lcfg.get("kernel_size_factor", 1.0))), 1)
        kernel += 1 - kernel % 2
        padding = same_padding(kernel, dilation, lcfg.get("future_context", -1))
        layers, c = [], cin
        for i in range(repeat):
            s = stride if i == repeat - 1 or not lcfg.get("stride_last", False) else 1
            if lcfg.get("separable", False):
                layers += [MaskedConv(c, c, kernel, s, dilation, padding, groups=c), MaskedConv(c, channels, 1)]
            else:
                layers.append(MaskedConv(c, channels, kernel, s, dilation, padding))
            layers.append(nn.BatchNorm1d(channels, eps=1e-3))
            if i < repeat - 1:
                layers += [activation, nn.Dropout(lcfg["dropout"])]
            c = channels
        if lcfg.get("se", False):
            layers.append(SqueezeExcite(channels, lcfg.get("se_reduction_ratio", 8), activation))
        self.mconv = nn.ModuleList(layers)
        self.res = None
        if lcfg["residual"]:
            res_stride = stride if residual_mode == "stride_add" else 1
            self.res = nn.ModuleList([nn.ModuleList([MaskedConv(cin, channels, 1, res_stride), nn.BatchNorm1d(channels, eps=1e-3)])])
        self.activation = activation
        self.dropout = nn.Dropout(lcfg["dropout"])

    def forward(self, x, lens):
        out, out_lens = x, lens
        for layer in self.mconv:
            if isinstance(layer, (MaskedConv, SqueezeExcite)):
                out, out_lens = layer(out, out_lens)
            else:
                out = layer(out)
        if self.res is not None:
            residual, _ = self.res[0][0](x, lens)
            out = out + self.res[0][1](residual)
        return self.dropout(self.activation(out)), out_lens


class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        require(cfg.get("_target_", "ConvASREncoder").rsplit(".", 1)[-1] == "ConvASREncoder", f"encoder {cfg.get('_target_')}")
        require(cfg.get("conv_mask", True), "conv_mask false")
        require(cfg.get("normalization_mode", "batch") == "batch", "normalization_mode")
        require(cfg.get("frame_splicing", 1) == 1, "frame_splicing")
        activation = ACTIVATIONS[cfg["activation"]]()
        blocks, cin = [], cfg["feat_in"]
        for lcfg in cfg["jasper"]:
            blocks.append(Block(cin, lcfg, activation, cfg.get("residual_mode", "add")))
            cin = lcfg["filters"]
        self.encoder = nn.ModuleList(blocks)

    def forward(self, x, lens):
        for block in self.encoder:
            x, lens = block(x, lens)
        return x, lens


class Citrinet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dec = cfg["decoder"]
        self.vocabulary = list(dec["vocabulary"])
        num_classes = dec.get("num_classes", -1)
        num_classes = len(self.vocabulary) if num_classes <= 0 else num_classes
        require(num_classes == len(self.vocabulary), "num_classes differs from vocabulary size")
        self.blank = num_classes
        self.features = Features(cfg["preprocessor"])
        self.encoder = Encoder(cfg["encoder"])
        self.decoder = nn.Module()
        self.decoder.decoder_layers = nn.ModuleList([nn.Conv1d(dec["feat_in"], num_classes + 1, 1)])

    @classmethod
    def from_nemo(cls, path, map_location="cpu"):
        with tarfile.open(path) as tar:
            members = {os.path.basename(m.name): m for m in tar.getmembers() if m.isfile()}
            cfg = yaml.safe_load(tar.extractfile(members["model_config.yaml"]))
            weights = io.BytesIO(tar.extractfile(members["model_weights.ckpt"]).read())
        state = torch.load(weights, map_location=map_location, weights_only=True)
        state = {k.replace("preprocessor.featurizer.", "features.", 1): v for k, v in state.items()}
        model = cls(cfg)
        model.load_state_dict(state, strict=True)
        return model.eval()

    def forward(self, audio, lens):
        with torch.backends.nnpack.flags(enabled=False):
            x, lens = self.features(audio, lens)
            x, lens = self.encoder(x, lens)
        logits = self.decoder.decoder_layers[0](x).transpose(1, 2)
        return torch.log_softmax(logits, dim=-1), lens

    def detokenize(self, ids):
        pieces = [self.vocabulary[i] for i in ids if not SPECIAL_TOKEN.fullmatch(self.vocabulary[i])]
        if any(t.startswith("##") for t in self.vocabulary):
            text = "".join(p[2:] if p.startswith("##") else p if "'" in (p, q) else " " + p for q, p in zip([""] + pieces, pieces))
        elif any(" " in t or "▁" in t for t in self.vocabulary):
            text = "".join(pieces)
        else:
            text = " ".join(pieces)
        return " ".join(text.replace("▁", " ").split())

    def greedy(self, log_probs, lens):
        texts = []
        for row, n in zip(log_probs.argmax(-1).cpu(), lens.tolist()):
            ids = torch.unique_consecutive(row[:n])
            texts.append(self.detokenize(ids[ids != self.blank].tolist()))
        return texts

    def load_audio(self, path):
        import soundfile

        audio, sr = soundfile.read(path, dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        target = self.features.sample_rate
        if sr != target:
            import numpy
            import soxr

            n = math.ceil(len(audio) * target / sr)
            audio = soxr.resample(audio, sr, target, "HQ")[:n]
            audio = numpy.pad(audio, (0, n - len(audio)))
        return torch.from_numpy(audio)

    @torch.inference_mode()
    def transcribe(self, paths, batch_size=8):
        training = self.training
        self.eval()
        device = self.features.fb.device
        audios = [self.load_audio(p) for p in paths]
        order = sorted(range(len(paths)), key=lambda i: len(audios[i]))
        texts = [None] * len(paths)
        for start in range(0, len(paths), batch_size):
            idx = order[start : start + batch_size]
            lens = torch.tensor([len(audios[i]) for i in idx], device=device)
            batch = nn.utils.rnn.pad_sequence([audios[i] for i in idx], batch_first=True).to(device)
            for i, text in zip(idx, self.greedy(*self(batch, lens))):
                texts[i] = text
        self.train(training)
        return texts


if __name__ == "__main__":
    model = Citrinet.from_nemo(sys.argv[1])
    for path, text in zip(sys.argv[2:], model.transcribe(sys.argv[2:])):
        print(f"{path}\t{text}")
