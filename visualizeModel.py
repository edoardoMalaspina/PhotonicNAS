import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
from nnabla.utils.save import save
from nnabla.utils.nnp_graph import NnpLoader

nnp = NnpLoader('log/classification/darts/cifar10/search/results.nnp')
net = nnp.get_network("classification/darts/cifar10/search")
list = net.get_network_names()
print(list)