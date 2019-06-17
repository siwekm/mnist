import numpy as np
import matplotlib.pyplot as plt

nCycles = 10            # number of training cycles
nTestSamples = 100      # number of test samples
nTrainSamples = 500     # number of training samples

class NNet:
    # NNet configuration
    # Number of Neurons
    # L0 - 28*28     -   INPUT
    # L1 - 64
    # L2 - 16
    # L3 - 10        -   OUTPUT


    def __init__(self,lr):
        # Learning rate
        self.lr = lr
        # Configuring synapses
        np.random.seed(1)
        self.syn0 = 2 * np.random.random((28 * 28, 64)) - 1
        self.syn1 = 2 * np.random.random((64, 16)) - 1
        self.syn2 = 2 * np.random.random((16, 10)) - 1

    def nonlin(x, deriv=False):
        if deriv is True:
            return x * (1 - x)
        else:
            return 1 / (1 + np.exp(-x))

    def train(self,x ,y):
        l0 = x
        l1 = self.nonlin(np.dot(l0, self.syn0))
        l2 = self.nonlin(np.dot(l1, self.syn1))
        l3 = self.nonlin(np.dot(l2, self.syn2))

        # setting result array
        target = np.array([0.01] * 10)
        target[y] = 0.99

        # err = (l3-res)**2

        # Calculating errors
        err_l3 = target - l3
        err_l2 = np.dot(self.syn2, err_l3)
        err_l1 = np.dot(self.syn1, err_l2)

        l3_delta = self.lr * err_l3 * self.nonlin(l3, True)
        l2_delta = self.lr * err_l2 * self.nonlin(l2, True)
        l1_delta = self.lr * err_l1 * self.nonlin(l1, True)

        # updating wieights of synapsis
        self.syn2 += np.dot(np.reshape(l2, (16, 1)), np.reshape(l3_delta, (1, 10)))
        self.syn1 += np.dot(np.reshape(l1, (64, 1)), np.reshape(l2_delta, (1, 16)))
        self.syn0 += np.dot(np.reshape(l0, (784, 1)), np.reshape(l1_delta, (1, 64)))

        target[y] = 0.01
        return err_l3[y]


    def run(self, x):
        l0 = x
        l1 = self.nonlin(np.dot(l0, self.syn0))
        l2 = self.nonlin(np.dot(l1, self.syn1))
        l3 = self.nonlin(np.dot(l2, self.syn2))
        return l3


network = NNet
network.__init__(network,0.2)
X = np.array([])  # Input values loaded from dataset
Y = np.array([])  # Output values loaded from dataset

stats_file = open("stats.txt", 'w')


# load dataset
data_file = open("mnist_test.csv", 'r')
data_list = data_file.readlines()
data_file.close()

# preparing training data
for i in range(nTrainSamples):
    values = data_list[i].split(',')
    scaled = (np.asfarray(values[1:])/ 255.0 * 0.99) + 0.01
    X = np.append(X, scaled)
    Y = np.append(Y,values[0])

X = np.reshape(X,(nTrainSamples,28*28))
Y = np.reshape(Y,(nTrainSamples,1))

# training NNet
for cycle in range(nCycles):
    err = 0
    for i in range(nTrainSamples):
        err += NNet.train(NNet,X[i],int(Y[i]))
        if i % 10 == 0:
            stats_file.write(str(err/10) + '\n')
            err = 0


X = np.array([])
Y = np.array([])

# load testing data
data_file = open("mnist_test_100.csv", 'r')
data_list = data_file.readlines()
data_file.close()

# preparing testing dataset
for i in range(nTestSamples):
    values = data_list[i].split(',')
    scaled = (np.asfarray(values[1:])/ 255.0 * 0.99) + 0.01
    X = np.append(X, scaled)
    Y = np.append(Y,values[0])

X = np.reshape(X,(nTestSamples,28*28))
Y = np.reshape(Y,(nTestSamples,1))

# testing trained NNet on different dataset
corr = 0
print("Guess | Real")
for i in range(nTestSamples):
    res = np.asfarray(NNet.run(NNet,X[i]))
    if res.argmax() == int(Y[i]):
        corr += 1

    image_array = np.asfarray(X[i]).reshape((28, 28))
    print(res.argmax(), int(Y[i]))
    plt.imshow(image_array, cmap='Greys', interpolation='None')
    # comment this line to skip
    plt.pause(1)

print("Score:", corr/nTestSamples * 100)
print("End")