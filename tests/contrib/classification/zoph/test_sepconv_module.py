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

import nnabla as nn

from nnabla_nas.contrib import zoph
from nnabla_nas.module import static as smo


def test_sepconv_module():
    shape = (10, 3, 32, 32)
    input = smo.Input(nn.Variable(shape))

    conv = zoph.SepConv(parents=[input],
                        in_channels=3,
                        out_channels=64,
                        kernel=(3, 3),
                        pad=(0, 0),
                        dilation=(1, 1),
                        with_bias=False
                        )

    assert conv.shape == (10, 64, 30, 30)


if __name__ == '__main__':
    test_sepconv_module()
