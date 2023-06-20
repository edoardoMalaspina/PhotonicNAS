
import nnabla.functions as F

from .module import Module


class PhotonicSigmoid(Module):
    def __init__(self, inplace=False, name=''):
        self._scope_name = f'<photonicsigmoid at {hex(id(self))}>'
        Module.__init__(self, name=name)
        self._inplace = inplace

    def call(self, input):
        tmp = F.exp((x - 0.145) / 0.073)
        result = 1.005 + (0.06 - 1.005) / (1 + tmp)
        return result

    def extra_repr(self):
        return f'inplace={self._inplace}'
