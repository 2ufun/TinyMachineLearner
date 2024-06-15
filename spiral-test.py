# Code for creating a spiral dataset from CS231n
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from Alpha.TinyLearner import *
from Alpha.helper import plot_decision_boundary, softmax

# %% prepare the data
N = 100  # number of points per class
D = 2  # dimensionality
K = 3 # number of classes
X = np.zeros((N * K, D))  # data matrix (each row = single example)
y = np.zeros(N * K, dtype='uint8')  # class labels
for j in range(K):
    ix = range(N * j, N * (j + 1))
    r = np.linspace(0.0, 1, N)  # radius
    np.random.seed(42)
    t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2  # theta
    X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
    y[ix] = j
# lets visualize the data
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.show()

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.4, random_state=42)

# %% build the neural network

class NN:
    def __init__(self):
        self.x = create_zeros(2, 1)
        self.weights_1 = create_randoms(20, 2, 42)
        self.bias_1 = create_randoms(20, 1, 42)

        # self.weights_2 = create_randoms(5, 5, 42)
        # self.bias_2 = create_randoms(5, 1, 42)

        self.weights_3 = create_randoms(3, 20, 42)
        self.bias_3 = create_randoms(3, 1, 42)

        tmp = mat_mul(self.weights_1, self.x)
        tmp = mat_add(self.bias_1, tmp)
        tmp = [[ReLU(i) for i in row] for row in tmp]
        # tmp = mat_mul(self.weights_2, tmp)
        # tmp = mat_add(self.bias_2, tmp)
        # tmp = [[ReLU(i) for i in row] for row in tmp]
        tmp = mat_mul(self.weights_3, tmp)
        self.y_pred = mat_add(self.bias_3, tmp)

        self.y_true = create_zeros(3, 1)

        self.params = []
        self.params.append(self.weights_1)
        self.params.append(self.bias_1)
        # self.params.append(self.weights_2)
        # self.params.append(self.bias_2)
        self.params.append(self.weights_3)
        self.params.append(self.bias_3)

    def __call__(self, x) -> list[float]:
        for i in range(len(x)):
            self.x[i][0].v = x[i]
        return [y[0].value() for y in self.y_pred]

    def fit(self, x, y):
        for i in range(len(x)):
            self.x[i][0].v = x[i]

        for i in range(len(self.y_true)):
            self.y_true[i][0].v = 0
        self.y_true[y][0].v = 1


nn = NN()
loss = CrossEntropyLoss(nn)
optimizer = Adam(nn.params, lr=0.01)

best_acc = 0
start = time.time()
for i in range(500):
    # train loop
    for ax, ay in zip(X_train, y_train):
        nn.fit(ax, ay)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # test loop
    y_preds = []
    for ax in X_test:
        y_logits = nn(ax)
        y_pred = np.argmax(softmax(y_logits))
        y_preds.append(y_pred)

    y_preds = np.array(y_preds)
    tf = np.sum(y_preds == y_test)
    acc = tf / len(y_preds)

    if acc > best_acc:
        best_acc = acc
        plot_decision_boundary(nn, X_test, y_test, 'images/spiral-boundary.png')

    print(f'[{i}] Accuracy: {acc:.2f}, Best: {best_acc:.2f}')

    if best_acc == 1:
        break

end = time.time()
print(f'Time taken: {end - start}')
