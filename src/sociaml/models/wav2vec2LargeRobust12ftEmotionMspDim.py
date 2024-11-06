'''
This script is used to load the pretrained model and process the input audio signal to get the embeddings or predict the emotions. Copied and adapted from https://huggingface.co/audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim
'''

import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)
from ..utils import get_device


class RegressionHead(nn.Module):
    r"""Classification head."""

    def __init__(self, config):

        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):

        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x


class EmotionModel(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""

    def __init__(self, config):

        super().__init__(config)

        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(
            self,
            input_values,
    ):

        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)

        return hidden_states, logits

class Wav2vec2LargeRobust12ftEmotionMspDim():
    def __init__(self, device = get_device()):
        self.device = device
        model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = EmotionModel.from_pretrained(model_name).to(device)

    def process(self, x: np.ndarray, sampling_rate: int, embeddings: bool = False) -> np.ndarray:
        r"""Predict emotions or extract embeddings from raw audio signal."""

        y = self.processor(x, sampling_rate=sampling_rate)
        y = y['input_values'][0]
        y = y.reshape(1, -1)
        y = torch.from_numpy(y).to(self.device)

        # run through model
        with torch.no_grad():
            y = self.model(y)[0 if embeddings else 1]

        # convert to numpy
        y = y.detach().cpu().numpy()

        return y

