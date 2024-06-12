import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from TinyLearner import *

# %% read data
df = pd.read_csv('data/iris.data', header=None)
X = np.array(df.iloc[:, 2:4])
name_dict = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
y = df.iloc[:, 4].map(name_dict).to_numpy()

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.4, random_state=42)

plt.scatter(x=X[:, 0],
            y=X[:, 1],
            c=y,
            cmap=plt.cm.RdYlBu)
plt.show()

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
        d = [Abs(Sub(self.y[i], self.net[i][0])) for i in range(len(self.y))]
        d = Add(*d)
        self.loss = Abs(d)

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


classifier = Classifier(Adam(lr=0.01))


# %% train model

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


for epoch in range(500):
    print('Epoch:', epoch)
    for ax, ay in zip(X_train, y_train):
        classifier.fit(ax, ay)


y_preds = []

for ax in X:
    y_logits = np.array(classifier(ax))
    y_pred = softmax(y_logits).argmax()
    y_preds.append(y_pred)

acc = np.sum(y_preds == y) / len(y)
print(f'Accuracy: {acc}')

plt.scatter(x=X[:, 0],
            y=X[:, 1],
            c=y_preds,
            cmap=plt.cm.RdYlBu)
plt.show()
