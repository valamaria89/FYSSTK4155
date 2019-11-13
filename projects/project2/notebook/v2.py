import random

# Third-party libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import sys

class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        print("SHAPES")
        print(len(self.biases))
        print(len(self.weights))
        for i in range(1, 3):
            print(self.biases[i-1].shape)
            print(self.weights[i-1].shape)
        self.weights = []
        self.biases = []
        for i in range(1,3):
            self.weights.append(np.load(f"w{i}.npy"))
            b = np.load(f"b{i}.npy")
            self.biases.append(np.reshape(b, (len(b), 1)))

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        self.losses = []
        for j in range(epochs):
            #random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                self.losses.append(self.evaluate(test_data))
                print("Epoch {0}: {1} / {2}".format(
                    j, self.losses[-1], n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            # 
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        #print(x)
        #print(y)
        #print("Activations")
        #for i, a in enumerate(activations):
        #   print(i, a)

        # backward pass
        print("Z")
        for z in zs:
            print(z)
            print()
        print("COST DERIVATIVE")
        print(self.cost_derivative(activations[-1], y) )
        print("SIGMOID DERIVATIVE")
        print(sigmoid_prime(zs[-1]))
        for i in [-1, -0.5, 0, 0.5, 1]:
            print(sigmoid_prime(i))
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        print("NABLA")
        print(nabla_b[-1])
        print(nabla_w[-1])
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        print("WEIGHTS")
        for w in nabla_w:
            print(w)
        sys.exit()
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def loss(self, data, labels):
        y_hat = self.feedforward(data)
        return -np.mean(labels*lognull(y_hat) + (1-labels)*lognull(1-y_hat))
        

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def lognull(x):
    return np.where(x <= 0, 0.0, np.log(x))

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

if __name__ == "__main__":
    observables, target = make_classification(n_samples=2000, n_features=25, n_informative=24, n_redundant=1, n_repeated=0, n_classes=2, random_state=42, flip_y=0.05, class_sep=1.0)
    X, Y = observables, target
    data = []
    test = []
    for c in range(X.shape[0]):
        x = np.reshape(X[c, :], (25, 1))
        y = np.array([0, 1]) if Y[c] else np.array([1, 0])
        test.append((x, Y[c]))
        y = np.reshape(y, (2, 1))
        data.append((x, y))
    print(data[0][0].shape, data[0][1].shape)
    X, y = np.load("X.npy"), np.load("Y.npy")
    labels = np.load("labels.npy")
    data = []
    test = []
    print(X.shape, y.shape, labels.shape)
    for c in range(X.shape[1]):
        x = np.reshape(X[:, c], (25, 1))
        y_ = np.reshape(y[:, c], (2, 1))
        data.append((x, y_))
        test.append((x, labels[0, c]))
    print(data[0][0].shape, data[0][1].shape)
    net = Network([25, 5, 2])
    net.SGD(data, 100, 10, 0.1, test_data=test)
    plt.plot(net.losses)
    plt.show()
