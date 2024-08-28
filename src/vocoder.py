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

"""HiFi-GAN vocoder."""

import torch
from speechbrain.lobes.models.HifiGAN import HifiganGenerator
from torch import nn


__all__ = ["HifiganVocoder"]


# Use default parameters from:
# https://github.com/bshall/knn-vc/blob/848302a262f7299c738af49d74209790ed442a9f/hifigan/config_v1_wavlm.json
class HifiganVocoder(HifiganGenerator):
    def __init__(
        self,
        embedding_dim,
        out_channels=1,
        resblock_type=1,
        resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
        resblock_kernel_sizes=(3, 7, 11),
        upsample_kernel_sizes=(20, 16, 4, 4),
        upsample_initial_channel=512,
        upsample_factors=(10, 8, 2, 2),
        inference_padding=5,
        cond_channels=0,
        conv_post_bias=True,
    ):
        if isinstance(embedding_dim, (list, tuple)):
            assert all([x == embedding_dim[0] for x in embedding_dim])
            self.embedding_dim = embedding_dim[0]
            self.num_codebooks = len(embedding_dim)
        else:
            self.embedding_dim = embedding_dim
            self.num_codebooks = 1
        super().__init__(
            in_channels=self.embedding_dim,
            out_channels=out_channels,
            resblock_type=str(resblock_type),
            resblock_dilation_sizes=resblock_dilation_sizes,
            resblock_kernel_sizes=resblock_kernel_sizes,
            upsample_kernel_sizes=upsample_kernel_sizes,
            upsample_initial_channel=upsample_initial_channel,
            upsample_factors=upsample_factors,
            inference_padding=inference_padding,
            cond_channels=cond_channels,
            conv_post_bias=conv_post_bias,
        )
        self.in_proj = nn.Linear(self.num_codebooks, 1)

    def forward(self, x, g=None):
        # (batch, time, embedding_dim, num_codebooks)
        x = self.in_proj(x)
        # (batch, time, embedding_dim, 1)
        x = x.squeeze(dim=-1)
        # (batch, time, embedding_dim)
        x = x.movedim(-1, -2)
        # (batch, embedding_dim, time)
        return super().forward(x, g)

    @torch.no_grad()
    def inference(self, x, g=None, **kwargs):
        return self.forward(x, g)


if __name__ == "__main__":
    from copy import deepcopy

    embedding_dim = 200
    x = torch.randn(2, 49, embedding_dim, 1)
    model = HifiganVocoder(embedding_dim)
    output = model(x)
    print(output.shape)
    with torch.no_grad():
        model(x)
        deepcopy(model)
