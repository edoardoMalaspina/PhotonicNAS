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


class Identity(Module):
    r"""Identity layer.
    A placeholder identity operator that is argument-insensitive.
    """
    def __init__(self, name=''):
        Module.__init__(self, name=name)
        self._scope_name = f'<identity at {hex(id(self))}>'

    def call(self, input):
        return input
