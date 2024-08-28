# ==============================================================================
# Copyright 2024 Luca Della Libera.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""PyTorch Hub entry point."""

import huggingface_hub
import torch
from safetensors import safe_open
from speechbrain.lobes.models.huggingface_transformers.wavlm import WavLM
from speechbrain.lobes.models.transformer.TransformerASR import TransformerASR

from src.codec import Codec
from src.dequantizer import Dequantizer
from src.quantizer import KMeansMultiQuantizer
from src.utils import SBWav2Vec2ForwardWrapper
from src.vocoder import HifiganVocoder


dependencies = [
    "huggingface_hub",
    "safetensors",
    "speechbrain",
    "torch",
    "transformers",
]


def discrete_wavlm_large(
    pretrained=True, cache_dir=huggingface_hub.constants.HUGGINGFACE_HUB_CACHE
) -> "Codec":
    """Load discrete WavLM codec.

    Arguments
    ---------
    pretrained:
        True to load the pretrained model weights, False otherwise.
    cache_dir:
        The model cache directory.

    """
    layer_ids = [6]
    encoder = WavLM(
        source="microsoft/wavlm-large",
        save_path=cache_dir,
        output_all_hiddens=True,
        output_norm=False,
    )
    encoder = SBWav2Vec2ForwardWrapper(encoder, layer_ids)

    num_features = 1024
    num_clusters = [512]
    quantizer = KMeansMultiQuantizer(num_features, num_clusters)

    dropout = 0.1
    activation = torch.nn.GELU
    d_model = 512
    nhead = 4
    num_layers = 6
    d_ffn = 512
    max_length = 2000
    causal = False
    dequantizer = Dequantizer(
        frontend=torch.nn.Linear(in_features=len(layer_ids), out_features=1),
        backbone=TransformerASR(
            input_size=num_features,
            tgt_vocab=-1,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=0,
            d_ffn=d_ffn,
            dropout=dropout,
            activation=activation,
            max_length=max_length,
            encoder_module="conformer",
            normalize_before=True,
            causal=causal,
        ),
        head=torch.nn.Linear(in_features=d_model, out_features=num_features),
        backend=torch.nn.Linear(in_features=1, out_features=len(layer_ids)),
    )

    resblock_type = 1
    resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    resblock_kernel_sizes = [3, 7, 11]
    upsample_kernel_sizes = [20, 16, 4, 4]
    upsample_initial_channel = 512
    upsample_factors = [10, 8, 2, 2]
    vocoder = HifiganVocoder(
        embedding_dim=[num_features],
        out_channels=1,
        resblock_type=str(resblock_type),
        resblock_dilation_sizes=resblock_dilation_sizes,
        resblock_kernel_sizes=resblock_kernel_sizes,
        upsample_kernel_sizes=upsample_kernel_sizes,
        upsample_initial_channel=upsample_initial_channel,
        upsample_factors=upsample_factors,
    )

    if pretrained:
        repo_id = "lucadellalib/discrete-wavlm-codec"
        for module, ckpt_file in zip(
            [quantizer, dequantizer, vocoder],
            ["quantizer.safetensors", "dequantizer.safetensors", "vocoder.safetensors"],
        ):
            local_path = huggingface_hub.hf_hub_download(
                repo_id, ckpt_file, cache_dir=cache_dir
            )
            with safe_open(local_path, framework="pt", device="cpu") as f:
                module.load_state_dict({k: f.get_tensor(k) for k in f.keys()})

    codec = Codec(encoder, quantizer, dequantizer, vocoder)
    codec.sample_rate = 16000

    return codec


if __name__ == "__main__":
    try:
        import torchaudio
    except ImportError:
        raise ImportError("`pip install torchaudio` to run this script")

    codec = discrete_wavlm_large(pretrained=True)
    print(
        f"Total number of parameters: {sum([x.numel() for x in codec.state_dict().values()]) / 1e6} M"
    )
    codec.eval().requires_grad_(False)
    sig, sample_rate = torchaudio.load("sample.wav")
    sig = torchaudio.functional.resample(sig, sample_rate, codec.sample_rate)
    feats = codec.sig_to_feats(sig)
    toks = codec.feats_to_toks(feats)
    qfeats = codec.toks_to_qfeats(toks)
    rec_feats = codec.qfeats_to_feats(qfeats)
    rec_sig = codec.feats_to_sig(rec_feats)
    torchaudio.save("reconstruction.wav", rec_sig[:, 0], codec.sample_rate)
