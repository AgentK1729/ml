from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer, SoftmaxLayer
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

from multithread import TrainerThread
import pickle

# Returns a network that has 2 input, 3 hidden and 2 output neurons
# net = buildNetwork(1, 3, 1, bias=True)
f1 = open("training.txt", "r")
net = pickle.load(f1)

# Dataset with two dimensional input and one dimensional target
ds = SupervisedDataSet(1, 1)

# Add data; simple even and odd classification
for i in range(1, 51):
    ds.addSample((i,), (i%2,))

trainer = BackpropTrainer(net, ds)

for i in range(100):
    print "Training session #%d" % (i+1)
    trainer.trainUntilConvergence()

f1 = open("training.txt", "w")
pickle.dump(net, f1)

for i in range(1, 11):
    print net.activate((i,))

"""threads = []
for i in range(3):
    threads.append(TrainerThread(trainer))

for thread in threads:
    thread.start()

for thread in threads:
    thread.join()"""
