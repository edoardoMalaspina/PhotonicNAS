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

from .module import Module


class Lambda(Module):
    r"""Lambda module.

    This module wraps a NNabla operator.

    Args:
        func (nnabla.functions): A NNabla funcion.
        name (string): the name of this module
    """

    def __init__(self, func, name=''):
        Module.__init__(self, name=name)
        self._scope_name = f'<{func.__name__} at {hex(id(self))}>'
        self._func = func

    def call(self, *args, **kargs):
        return self._func(*args, **kargs)

    def extra_repr(self):
        return f'function={self._func.__name__}'
