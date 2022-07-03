#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 20:07:19 2022

@author: atreyee
"""
import torch
from torch import Tensor
from torch import nn



class Wav2Letter(nn.Module):
    r"""Wav2Letter model architecture from *Wav2Letter: an End-to-End ConvNet-based Speech
    Recognition System* [:footcite:`collobert2016wav2letter`].

     :math:`\text{padding} = \frac{\text{ceil}(\text{kernel} - \text{stride})}{2}`

    Args:
        num_classes (int, optional): Number of classes to be classified. (Default: ``40``)
        input_type (str, optional): Wav2Letter can use as input: ``waveform``, ``power_spectrum``
         or ``mfcc`` (Default: ``waveform``).
        num_features (int, optional): Number of input features that the network will receive (Default: ``1``).
    """

    def __init__(self, num_classes: int = 40, input_type: str = "wav2vec", num_features: int = 1, max_len: int = 250):
        super(Wav2Letter, self).__init__()

        acoustic_num_features = 250 if input_type == "waveform" else num_features
        acoustic_model = nn.Sequential(
            nn.Conv1d(in_channels=acoustic_num_features, out_channels=250, kernel_size=48, stride=2, padding=23),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=250, out_channels=250, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=250, out_channels=250, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=250, out_channels=250, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=250, out_channels=250, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=250, out_channels=250, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=250, out_channels=250, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=250, out_channels=250, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=250, out_channels=2000, kernel_size=32, stride=1, padding=16),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=2000, out_channels=2000, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=2000, out_channels=num_classes, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )

        assert input_type in ["wavform", "power_spectrum", "mfcc", "wav2vec"], "Unrecognized input_type."

        if input_type == "waveform":
            waveform_model = nn.Sequential(
                nn.Conv1d(in_channels=num_features, out_channels=250, kernel_size=250, stride=160, padding=45),
                nn.ReLU(inplace=True),
            )
            self.acoustic_model = nn.Sequential(waveform_model, acoustic_model)

        elif input_type in ["power_spectrum", "mfcc"]:
            self.acoustic_model = acoustic_model

        elif input_type == "wav2vec":
            self.acoustic_model = acoustic_model

        self.conv_layer_properties = self.get_sequential_model_properties(self.acoustic_model)
        self.num_conv_layers = len(self.conv_layer_properties)
        self.max_len = max_len

    def get_sequential_model_properties(self, model):

        kernel_size = []; stride = []; padding = []; dilation = []
        for i, module in enumerate(model.modules()):

            if isinstance(module, nn.Conv1d):
                kernel_size.append(module.kernel_size[0])
                stride.append(module.stride[0])
                padding.append(module.padding[0])
                dilation.append(module.dilation[0])

        return dict({'kernel_size': kernel_size, 'stride': stride, 'padding': padding, 'dilation': dilation})

    def get_conv1d_outlens(self, src_lengths, kernel_size, stride, padding, dilation):
        for i in range(self.num_conv_layers):
            src_lengths = (src_lengths+2*padding[i]-dilation[i]*(kernel_size[i]-1)-1)/stride[i] + 1

        return src_lengths.to(torch.int32)

    def make_pad_mask(self, lengths, xs=None, length_dim=-1):
        if not isinstance(lengths, list):
            lengths = lengths.tolist()
        bs = int(len(lengths))

        seq_range = torch.arange(0, self.max_len, dtype=torch.int64)
        seq_range_expand = seq_range.unsqueeze(0).expand(bs, self.max_len)
        seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
        mask = seq_range_expand >= seq_length_expand

        if xs is not None:
            assert xs.size(0) == bs, (xs.size(0), bs)

            if length_dim < 0:
                length_dim = xs.dim() + length_dim
            # ind = (:, None, ..., None, :, , None, ..., None)
            ind = tuple(
                slice(None) if i in (0, length_dim) else None for i in range(xs.dim())
            )
            # mask = mask[ind].expand_as(xs).to(xs.device)
            mask = mask[ind].to(xs.device)
        return mask

    def forward(self, x, src_lengths):
        r"""
        Args:
            x (torch.Tensor): Tensor of dimension (batch_size, num_features, input_length).

        Returns:
            Tensor: Predictor tensor of dimension (batch_size, number_of_classes, input_length).
        """

        x = self.acoustic_model(x)
        x = nn.functional.log_softmax(x.float(), dim=1)

        out_lengths = self.get_conv1d_outlens(src_lengths, **self.conv_layer_properties)
        mask = self.make_pad_mask(out_lengths, x)

        return x, mask
