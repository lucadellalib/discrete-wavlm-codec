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

"""Neural dequantizer."""

from typing import Optional

import torch
from torch import Tensor, nn


__all__ = ["Dequantizer"]


class Dequantizer(nn.Module):
    """Dequantizer.

    Arguments
    ---------
    backbone:
        The transformer backbone.
    embedding:
        The transformer embedding layer.
    frontend:
        The transformer frontend.
    head:
        The transformer head.

    Examples
    --------
    >>> from speechbrain.lobes.models.convolution import ConvolutionFrontEnd
    >>> from speechbrain.lobes.models.transformer.TransformerASR import TransformerASR
    >>> from torch import nn
    >>>
    >>> input_size = 256
    >>> d_model = 128
    >>> out_channels = (28, 28, 28)
    >>> strides = [1, 2, 2]
    >>> frontend = ConvolutionFrontEnd((None, None, input_size), out_channels=out_channels, strides=strides)
    >>> backbone = TransformerASR(
    ...     input_size=input_size // torch.Size(strides).numel() * out_channels[-1],
    ...     tgt_vocab=-1,
    ...     num_decoder_layers=0,
    ...     d_model=d_model,
    ... )
    >>> head = nn.Linear(d_model, input_size)
    >>> model = Dequantizer(backbone, frontend=frontend, head=head)
    >>>
    >>> input = torch.rand([10, 200, input_size])
    >>> length = torch.ones(10)
    >>> output = model(input, length)

    """

    def __init__(
        self,
        backbone: "nn.Module",
        embedding: "Optional[nn.Module]" = None,
        frontend: "Optional[nn.Module]" = None,
        head: "Optional[nn.Module]" = None,
        backend: "Optional[nn.Module]" = None,
    ) -> "None":
        super().__init__()
        self.backbone = backbone
        self.embedding = embedding
        self.frontend = frontend
        self.head = head
        self.backend = backend

    def forward(self, src: "Tensor", length: "Optional[Tensor]" = None) -> "Tensor":
        if self.embedding is not None:
            src = self.embedding(src)

        if self.frontend is not None:
            src = self.frontend(src)

        src_shape = src.shape
        if len(src_shape) > 3:
            # assert src_shape[-1] == 1
            src = src.squeeze(dim=-1)

        if hasattr(self.backbone, "encode"):
            # Transformer ASR
            src = self.backbone.encode(src, length)
        else:
            src = self.backbone(src, length)
        if self.head is not None:
            src = self.head(src)

        if len(src_shape) > 3:
            src = src.unsqueeze(dim=-1)

        if self.backend is not None:
            src = self.backend(src)

        return src


if __name__ == "__main__":
    import torch
    from speechbrain.lobes.models.convolution import ConvolutionFrontEnd
    from speechbrain.lobes.models.transformer.TransformerASR import TransformerASR
    from torch import nn

    input_size = 256
    d_model = 128
    out_channels = (28, 28, 28)
    strides = [1, 2, 2]

    frontend = ConvolutionFrontEnd(
        (None, None, input_size), out_channels=out_channels, strides=strides
    )

    backbone = TransformerASR(
        input_size=input_size // torch.Size(strides).numel() * out_channels[-1],
        tgt_vocab=-1,
        num_decoder_layers=0,
        d_model=d_model,
    )

    head = nn.Linear(d_model, input_size)

    model = Dequantizer(backbone, frontend=frontend, head=head)

    input = torch.rand([10, 200, input_size])
    length = torch.ones(10)
    output = model(input, length)
