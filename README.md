# Discrete WavLM Codec

A speech codec obtained by quantizing WavLM representations via K-means clustering (see https://arxiv.org/abs/2312.09747).

---------------------------------------------------------------------------------------------------------

## 🛠️️ Installation

First of all, install [Python 3.8 or later](https://www.python.org). Open a terminal and run:

```
pip install huggingface-hub safetensors speechbrain torch torchaudio transformers
```

---------------------------------------------------------------------------------------------------------

## ▶️ Quickstart

We use `torch.hub` to make loading the model easy (no need to clone the repository):

```python
import torch
import torchaudio

dwavlm = torch.hub.load("lucadellalib/discrete-wavlm-codec", "discrete_wavlm_large", layer_ids=[6])
dwavlm.eval().requires_grad_(False)
sig, sample_rate = torchaudio.load("<path-to-audio-file>")
sig = torchaudio.functional.resample(sig, sample_rate, dwavlm.sample_rate)
feats = dwavlm.sig_to_feats(sig)
toks = dwavlm.feats_to_toks(feats)
qfeats = dwavlm.toks_to_qfeats(toks)
rec_feats = dwavlm.qfeats_to_feats(qfeats)
rec_sig = dwavlm.feats_to_sig(rec_feats)
torchaudio.save("reconstruction.wav", rec_sig[:, 0], dwavlm.sample_rate)
```

---------------------------------------------------------------------------------------------------------

## 📧 Contact

[luca.dellalib@gmail.com](mailto:luca.dellalib@gmail.com)

---------------------------------------------------------------------------------------------------------
