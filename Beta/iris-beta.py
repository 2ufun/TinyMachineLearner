import time

import pandas as pd
from sklearn.model_selection import train_test_split

from TinyLearnerBeta import *
from helper import plot_decision_boundary, softmax_for_network

# %% read data
df = pd.read_csv('../data/iris.data', header=None)
X = np.array(df.iloc[:, 2:4])
name_dict = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
y = df.iloc[:, 4].map(name_dict).to_numpy()

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.4, random_state=42)


# %% build neural network
class Classifier:
    def __init__(self, optimum):
        self.x = [[Number(0)], [Number(0)]]

        self.weights_from_input = create_randoms(3, 2, seed=42)
        self.bias_from_input = create_randoms(3, 1, seed=42)

        self.weights_hidden = create_randoms(3, 3, seed=42)
        self.bias_hidden = create_randoms(3, 1, seed=42)

        self.weights_to_output = create_randoms(3, 3, seed=42)
        self.bias_to_output = create_randoms(3, 1, seed=42)

        tmp = mat_mul(self.weights_from_input, self.x)
        tmp = mat_add(self.bias_from_input, tmp)
        tmp = [[ReLU(i) for i in row] for row in tmp]

        tmp = mat_mul(self.weights_hidden, tmp)
        tmp = mat_add(self.bias_hidden, tmp)
        tmp = [[ReLU(i) for i in row] for row in tmp]

        tmp = mat_mul(self.weights_to_output, tmp)
        self.net = mat_add(self.bias_to_output, tmp)

        self.y = [Number(0), Number(0), Number(0)]

        prob = softmax_for_network([n[0] for n in self.net])
        product = [Mul(self.y[i], Log(prob[i])) for i in range(len(self.y))]
        self.loss = Neg(Add(*product))

        self.optimum = optimum
        self.optimum.set_params([self.weights_from_input,
                                 self.bias_from_input,
                                 self.weights_hidden,
                                 self.bias_hidden,
                                 self.weights_to_output,
                                 self.bias_to_output])

    def __call__(self, x) -> list:
        for i in range(len(x)):
            self.x[i][0].set_value(x[i])
        return [[i.value() for i in row] for row in self.net]

    def fit(self, x, y):
        for i in range(len(x)):
            self.x[i][0].set_value(x[i])

        for i in range(len(self.y)):
            self.y[i].set_value(0)
        self.y[y].set_value(1)

        self.optimum.step(self.loss)


classifier = Classifier(SGD(lr=0.01))

# %% train model

start = time.time()
for i in range(10):
    for ax, ay in zip(X_train, y_train):
        classifier.fit(ax, ay)
end = time.time()

plot_decision_boundary(classifier, X_test, y_test, '../beta_boundary.png')

print(f'time: {end - start}')
