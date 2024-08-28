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

"""Common utilities."""

from typing import Optional, Sequence

import torch
from torch import Tensor, nn
from transformers.models.hubert.modeling_hubert import HubertEncoderStableLayerNorm
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2EncoderStableLayerNorm,
)
from transformers.models.wavlm.modeling_wavlm import WavLMEncoderStableLayerNorm


__all__ = ["SBWav2Vec2ForwardWrapper"]


class SBWav2Vec2ForwardWrapper(nn.Module):
    """SpeechBrain wav2vec 2.0 wrapper that returns the hidden representations from the specified layer IDs.

    Arguments
    ---------
    wav2vec2:
        The SpeechBrain wav2vec 2.0 module.
    layer_ids:
        The layer IDs from which the hidden representations are extracted.

    Examples
    --------
    >>> import torch
    >>> from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
    >>> from speechbrain.lobes.models.huggingface_transformers.wavlm import WavLM
    >>>
    >>> encoder = WavLM(source="microsoft/wavlm-large", save_path=HUGGINGFACE_HUB_CACHE)
    >>> encoder = SBWav2Vec2ForwardWrapper(encoder, layer_ids=[6, 7])
    >>>
    >>> input = torch.rand([10, 16000])
    >>> length = torch.ones(10)
    >>> output = encoder(input, length)

    """

    def __init__(self, wav2vec2: "nn.Module", layer_ids: "Sequence[int]") -> "None":
        super().__init__()
        self.wav2vec2 = wav2vec2
        # Workaround to deal with hardcoded class name in discrete SSL
        # https://github.com/speechbrain/speechbrain/blob/60062c2536e8122253d6ad0e681208f554528950/speechbrain/lobes/models/huggingface_transformers/discrete_ssl.py#L88
        self.__class__.__name__ = self.wav2vec2.__class__.__name__
        self.layer_ids = sorted(layer_ids)
        assert hasattr(self.wav2vec2, "model")
        assert hasattr(self.wav2vec2.model, "encoder")
        assert hasattr(self.wav2vec2.model.encoder, "layers")
        # Workaround for early exiting to avoid the computational overhead of forwarding through the whole model
        # NOTE: the model is modified in-place
        self.wav2vec2.output_all_hiddens = True
        self.wav2vec2.model.encoder.layers = self.wav2vec2.model.encoder.layers[
            : max(self.layer_ids)
        ]
        # NOTE: workaround to account for layer norm applied to the last hidden states when StableLayerNorm variant is used:
        # https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/models/wavlm/modeling_wavlm.py#L816
        if isinstance(
            self.wav2vec2.model.encoder,
            (
                HubertEncoderStableLayerNorm,
                Wav2Vec2EncoderStableLayerNorm,
                WavLMEncoderStableLayerNorm,
            ),
        ):
            self.wav2vec2.model.encoder.layer_norm = torch.nn.Identity()

    def extract_features(
        self, wav: "Tensor", length: "Optional[Tensor]" = None
    ) -> "Tensor":
        feats = self.wav2vec2(wav, length)  # (K, B, N, H)
        return feats[self.layer_ids]

    def forward(self, wav: "Tensor", length: "Optional[Tensor]" = None) -> "Tensor":
        return self.extract_features(wav, length)


if __name__ == "__main__":
    from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
    from speechbrain.lobes.models.huggingface_transformers.wav2vec2 import Wav2Vec2

    for source in [
        "facebook/wav2vec2-large-960h-lv60-self",
        "facebook/hubert-large-ll60k",
        "microsoft/wavlm-large",
    ]:
        layer_ids = [3, 7]
        encoder1 = Wav2Vec2(
            source=source,
            save_path=HUGGINGFACE_HUB_CACHE,
            output_norm=True,
        )
        encoder1 = SBWav2Vec2ForwardWrapper(encoder1, layer_ids=layer_ids).eval()

        encoder2 = Wav2Vec2(
            source=source,
            save_path=HUGGINGFACE_HUB_CACHE,
            output_norm=True,
            output_all_hiddens=True,
        ).eval()

        input = torch.ones([1, 16000])
        with torch.no_grad():
            output1 = encoder1(input)
            output2 = encoder2(input)[layer_ids]

        print((output1 == output2).all())
