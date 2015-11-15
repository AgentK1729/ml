from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer, SoftmaxLayer
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

from multithread import TrainerThread
import pickle

# Returns a network that has 1 input, 3 hidden and 1 output neurons
net = buildNetwork(1, 3, 1, bias=True)

# Dataset with two dimensional input and one dimensional target
ds = SupervisedDataSet(1, 1)

# Add data; simple even and odd classification
for i in range(1, 51):
    ds.addSample((i,), (i%2,))

trainer = BackpropTrainer(net, ds, learningrate=3, momentum=0.99)

for i in range(10):
    trainer.train()

f1 = open("training.txt", "w")
pickle.dump(net, f1)

for i in range(1, 11):
    print net.activate((i,))
