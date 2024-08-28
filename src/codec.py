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

"""Neural codec."""

from typing import Optional, Sequence, Union

from torch import Tensor, nn


__all__ = ["Codec"]


class Codec(nn.Module):
    """Neural codec.

    Arguments
    ---------
    encoder:
        The encoder, i.e. a module that receives as an input a waveform and returns
        the corresponding continuous hidden representations.
    quantizer:
        The quantizer, i.e. a module that receives as an input continuous hidden representations
        and returns the corresponding tokens and quantized hidden representations.
    dequantizer:
        The dequantizer, i.e. a module that receives as an input quantized hidden representations
        and returns the corresponding continuous hidden representations.
    vocoder:
        The vocoder, i.e. a module that receives as an input continuous hidden representations
        and returns the corresponding waveform.
    freeze:
        The names of the modules to freeze (e.g. `["encoder", "vocoder"]`).

    """

    def __init__(
        self,
        encoder: "Optional[nn.Module]" = None,
        quantizer: "Optional[nn.Module]" = None,
        dequantizer: "Optional[nn.Module]" = None,
        vocoder: "Optional[nn.Module]" = None,
        freeze: "Union[Sequence[str], bool]" = False,
    ) -> "None":
        super().__init__()
        self.encoder = encoder
        self.quantizer = quantizer
        self.dequantizer = dequantizer
        self.vocoder = vocoder
        self.freeze = freeze
        if isinstance(freeze, bool):
            if freeze:
                self.requires_grad_(False).eval()
            return
        for key in self.freeze:
            self._modules[key].requires_grad_(False).eval()

    def forward(self, sig: "Tensor", length: "Optional[Tensor]" = None) -> "Tensor":
        """Forward pass.

        Arguments
        ---------
        sig:
            The input waveform, shape: [B, T].
        length:
            The relative length, shape: [B].

        Returns
        -------
            The reconstructed waveform, shape (B, T).

        """
        feats = self.sig_to_feats(sig, length)
        qfeats = self.feats_to_qfeats(feats)
        rec_feats = self.qfeats_to_feats(qfeats, length)
        rec_sig = self.feats_to_sig(rec_feats)
        return rec_sig

    def sig_to_feats(
        self, sig: "Tensor", length: "Optional[Tensor]" = None
    ) -> "Tensor":
        if self.encoder is None:
            raise NotImplementedError
        feats = self.encoder(sig, length)  # (K, B, N, H)
        return feats.movedim(0, -1)  # (B, N, H, K)

    def feats_to_sig(self, feats: "Tensor") -> "Tensor":
        if self.vocoder is None:
            raise NotImplementedError
        # (B, N, H, K)
        sig = self.vocoder(feats)
        return sig  # (B, C, T)

    def feats_to_toks(self, feats: "Tensor") -> "Tensor":
        if self.quantizer is None:
            raise NotImplementedError
        # (B, N, H, K)
        toks, _ = self.quantizer(feats)
        return toks  # (B, N, K)

    def feats_to_qfeats(self, feats: "Tensor") -> "Tensor":
        if self.quantizer is None:
            raise NotImplementedError
        # (B, N, H, K)
        _, qfeats = self.quantizer(feats)
        return qfeats  # (B, N, H, K)

    def qfeats_to_feats(
        self, qfeats: "Tensor", length: "Optional[Tensor]" = None
    ) -> "Tensor":
        if self.dequantizer is None:
            raise NotImplementedError
        # (B, N, H, K)
        feats = self.dequantizer(qfeats, length)
        return feats  # (B, N, H, K)

    def toks_to_qfeats(self, toks: "Tensor") -> "Tensor":
        if self.quantizer is None:
            raise NotImplementedError
        # (B, N, K)
        _, qfeats = self.quantizer(toks)
        return qfeats  # (B, N, H, K)


if __name__ == "__main__":
    import torch
    from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
    from speechbrain.lobes.models.huggingface_transformers.wavlm import WavLM
    from speechbrain.lobes.models.transformer.TransformerASR import TransformerASR

    from dequantizer import Dequantizer
    from quantizer import KMeansMultiQuantizer
    from utils import SBWav2Vec2ForwardWrapper
    from vocoder import HifiganVocoder

    layer_ids = [6, 7]
    num_features = 768
    num_clusters = [300, 300]
    encoder = WavLM(
        source="microsoft/wavlm-base",
        save_path=HUGGINGFACE_HUB_CACHE,
        output_all_hiddens=True,
    )
    quantizer = KMeansMultiQuantizer(num_features, num_clusters)
    dequantizer = Dequantizer(
        frontend=torch.nn.Linear(in_features=len(layer_ids), out_features=1),
        backbone=TransformerASR(
            input_size=num_features,
            tgt_vocab=-1,
            d_model=128,
            nhead=4,
            num_encoder_layers=6,
            num_decoder_layers=0,
            d_ffn=512,
        ),
        head=torch.nn.Linear(in_features=128, out_features=num_features),
        backend=torch.nn.Linear(in_features=1, out_features=len(layer_ids)),
    )
    vocoder = HifiganVocoder(
        embedding_dim=[num_features] * len(layer_ids),
        out_channels=1,
        resblock_type="1",
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        resblock_kernel_sizes=[3, 7, 11],
        upsample_kernel_sizes=[16, 16, 4, 4],
        upsample_initial_channel=512,
        upsample_factors=[8, 8, 2, 2],
    )
    codec = Codec(
        SBWav2Vec2ForwardWrapper(encoder, layer_ids), quantizer, dequantizer, vocoder
    )
    sigs = torch.rand([10, 16000])
    rec_sig = codec(sigs)
    print(rec_sig.shape)
