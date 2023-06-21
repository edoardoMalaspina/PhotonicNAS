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

import nnabla.functions as F

from .module import Module


class PhotonicSigmoid(Module):
    def __init__(self, inplace=False, name=''):
        self._scope_name = f'<photonicsigmoid at {hex(id(self))}>'
        Module.__init__(self, name=name)
        self._inplace = inplace

    def call(self, input):
        result = 1.005 + (0.06 - 1.005) / (1. + F.exp((input - 0.145) / 0.073))
        return result

    def extra_repr(self):
        return f'inplace={self._inplace}'
