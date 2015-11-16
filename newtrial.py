from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer, SoftmaxLayer
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
import pickle
from random import randint

net = buildNetwork(1, 3, 1, bias=True)

# Dataset with two dimensional input and one dimensional target
ds = SupervisedDataSet(1, 1)

# Add data; simple even and odd classification
for i in range(1, 1001):
    ds.addSample((i,), (i%2,))


trainer = BackpropTrainer(net, ds, learningrate=30, momentum=0.99)

print randint(1,1000), net.activate((randint(1,1000),))
