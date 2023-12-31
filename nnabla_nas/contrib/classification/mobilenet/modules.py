# Copyright (c) 2020 Sony Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import OrderedDict

from .... import module as Mo

CANDIDATES = OrderedDict([
    ('MB1 3x3',
        lambda inc, outc, s, n: InvertedResidual(
            inc, outc, s, expand_ratio=1, kernel=(3, 3), name=n)),
    ('MB3 3x3',
        lambda inc, outc, s, n: InvertedResidual(
            inc, outc, s, expand_ratio=3, kernel=(3, 3), name=n)),
    ('MB6 3x3',
        lambda inc, outc, s, n: InvertedResidual(
            inc, outc, s, expand_ratio=6, kernel=(3, 3), name=n)),
    ('MB1 5x5',
        lambda inc, outc, s, n: InvertedResidual(
            inc, outc, s, expand_ratio=1, kernel=(5, 5), name=n)),
    ('MB3 5x5',
        lambda inc, outc, s, n: InvertedResidual(
            inc, outc, s, expand_ratio=3, kernel=(5, 5), name=n)),
    ('MB6 5x5',
        lambda inc, outc, s, n: InvertedResidual(
            inc, outc, s, expand_ratio=6, kernel=(5, 5), name=n)),
    ('MB1 7x7',
        lambda inc, outc, s, n: InvertedResidual(
            inc, outc, s, expand_ratio=1, kernel=(7, 7), name=n)),
    ('MB3 7x7',
        lambda inc, outc, s, n: InvertedResidual(
            inc, outc, s, expand_ratio=3, kernel=(7, 7), name=n)),
    ('MB6 7x7',
        lambda inc, outc, s, n: InvertedResidual(
            inc, outc, s, expand_ratio=6, kernel=(7, 7), name=n)),
    ('skip_connect',
        lambda inc, outc, s, n: Mo.Identity(name=n))
])


class ConvBNReLU(Mo.Sequential):
    r"""Convolution-BatchNormalization-ReLU layer.

    Args:
        in_channels (:obj:`int`): Number of convolution kernels (which is
            equal to the number of input channels).
        out_channels (:obj:`int`): Number of convolution kernels (which is
            equal to the number of output channels). For example, to apply
            convolution on an input with 16 types of filters, specify 16.
        kernel (:obj:`tuple` of :obj:`int`): Convolution kernel size. For
            example, to apply convolution on an image with a 3 (height) by 5
            (width) two-dimensional kernel, specify (3,5).
        stride (:obj:`tuple` of :obj:`int`, optional): Stride sizes for
            dimensions. Defaults to None.
    """

    def __init__(self, in_channels, out_channels, kernel=(3, 3),
                 stride=(1, 1), group=1, name=''):
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel = kernel
        self._stride = stride
        self._pad = ((kernel[0] - 1)//2, (kernel[1] - 1)//2)
        self._name = name

        super(ConvBNReLU, self).__init__(
            Mo.Conv(in_channels, out_channels, self._kernel,
                    stride=self._stride, pad=self._pad, group=group,
                    with_bias=False, name='{}/conv'.format(self.name)),
            Mo.BatchNormalization(n_features=out_channels, n_dims=4,
                                  name='{}/bn'.format(self.name)),
            Mo.ReLU6(name='{}/relu6'.format(self.name))
        )

    def extra_repr(self):
        return (f'in_channels={self._in_channels}, '
                f'out_channels={self._out_channels}, '
                f'kernel={self._kernel}, '
                f'stride={self._stride}, '
                f'pad={self._pad}')


class InvertedResidual(Mo.Module):
    """The Inverted-Resisual layer.

    Args:
        in_channels (:obj:`int`): Number of convolution kernels (which is
            equal to the number of input channels).
        out_channels (:obj:`int`): Number of convolution kernels (which is
            equal to the number of output channels). For example, to apply
            convolution on an input with 16 types of filters, specify 16.
        stride (:obj:`tuple` of :obj:`int`, optional): Stride sizes for
            dimensions. Defaults to None.
        kernel (:obj:`tuple` of :obj:`int`): Convolution kernel size. For
            example, to apply convolution on an image with a 3 (height) by 5
            (width) two-dimensional kernel, specify (3, 5).
        expand_ratio(:obj:`int`): The expand ratio.
    """

    def __init__(self, in_channels, out_channels, stride, kernel=(3, 3),
                 expand_ratio=1, name=''):

        assert stride in [1, 2]

        self._stride = stride
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel = kernel
        self._expand_ratio = expand_ratio
        self._name = name

        hidden_dim = int(round(in_channels * expand_ratio))
        self._use_res_connect = (self._stride == 1 and
                                 in_channels == out_channels)
        if self._use_res_connect:
            self._add_res = Mo.Add2(name='{}/add2'.format(self.name))

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(in_channels, hidden_dim, (1, 1),
                                     name='{}/ConvBNReLU_0'.format(self.name)))

        layers.extend([
            ConvBNReLU(hidden_dim, hidden_dim, kernel=kernel,
                       stride=(stride, stride), group=hidden_dim,
                       name='{}/ConvBNReLU_1'.format(self.name)),
            Mo.Conv(hidden_dim, out_channels, kernel=(1, 1), stride=(1, 1),
                    with_bias=False, name='{}/conv'.format(self.name)),
            Mo.BatchNormalization(n_features=out_channels, n_dims=4,
                                  name='{}/bn'.format(self.name))
        ])

        self._conv = Mo.Sequential(*layers)

    def call(self, x):
        if self._use_res_connect:
            # return x + self._conv(x)
            return self._add_res(x, self._conv(x))
        return self._conv(x)

    def extra_repr(self):
        return (f'in_channels={self._in_channels}, '
                f'out_channels={self._out_channels}, '
                f'kernel={self._kernel}, '
                f'stride={self._stride}, '
                f'expand_ratio={self._expand_ratio}')


class ChoiceBlock(Mo.Module):
    def __init__(self, in_channels, out_channels, stride,
                 ops, mode='sample', name=''):
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._stride = stride
        self._mode = mode

        self._mixed = Mo.MixedOp(
            operators=[CANDIDATES[k](in_channels, out_channels,
                                     stride, name)
                       for k in ops],
            mode=mode, name=name
        )

    def call(self, input):
        return self._mixed(input)

    def extra_repr(self):
        return (f'in_channels={self._in_channels}, '
                f'out_channels={self._out_channels}, '
                f'stride={self._stride}, '
                f'mode={self._mode}')
