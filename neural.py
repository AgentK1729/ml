from pybrain.structure import FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection

net = FeedForwardNetwork()

inLayer = LinearLayer(1)
hiddenLayer = SigmoidLayer(3)
outLayer = LinearLayer(1)

net.addInputModule(inLayer)
net.addModule(hiddenLayer)
net.addOutputModule(outLayer)

in_to_hidden = FullConnection(inLayer, hiddenLayer)
hidden_to_out = FullConnection(hiddenLayer, outLayer)

net.addConnection(in_to_hidden)
net.addConnection(hidden_to_out)

net.sortModules()

print net
