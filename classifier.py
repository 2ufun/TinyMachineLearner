import pickle

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from TinyLearner import *
from helper import plot_decision_boundary, softmax

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

        prob = Softmax([n[0] for n in self.net])
        prod = [Mul(self.y[i], Log(prob[i])) for i in range(len(self.y))]
        self.loss = Neg(Add(*prod))

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
best_acc = 0
loss_record = []
acc_record = []
for epoch in range(50):
    for ax, ay in zip(X_train, y_train):
        classifier.fit(ax, ay)

    y_logits = np.array([classifier(ax) for ax in X_test])
    y_pred = np.array([softmax(ay).argmax() for ay in y_logits])
    acc = np.sum(y_pred == y_test) / len(y_test)

    acc_record.append(acc)
    loss_record.append(classifier.loss.value())

    if acc > best_acc:
        best_acc = acc
        plot_decision_boundary(classifier, X, y)

    print(f'[{epoch}] Accuracy: {acc}, Best Accuracy: {best_acc}')

with open('acc_record.bin', 'wb') as file:
    pickle.dump(acc_record, file)

with open('loss_record.bin', 'wb') as file:
    pickle.dump(loss_record, file)

plt.plot(loss_record)
plt.title('Loss Figure')
plt.grid(True)
plt.savefig('images/loss_figure.png')
plt.show()

plt.plot(acc_record)
plt.title('Accuracy Figure')
plt.grid(True)
plt.savefig('images/accuracy_figure.png')
plt.show()
