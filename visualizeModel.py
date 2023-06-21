import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
from nnabla.utils.save import save
from nnabla.utils.nnp_graph import NnpLoader

nnp = NnpLoader('log/classification/darts/cifar10/search/results.nnp')

net = nnp.get_network(nnp.get_network_names()[0])

for i in net.variables.keys():
   print(i)