# read and split datasets
import time

import pandas as pd
from sklearn.model_selection import train_test_split

from Alpha.TinyLearner import *
from Alpha.helper import softmax, plot_decision_boundary

df = pd.read_csv('data/iris.data', header=None)
xs = np.array(df.iloc[:, 2:4])
name_dict = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
ys = df.iloc[:, 4].map(name_dict).to_numpy()

X_train, X_test, y_train, y_test = \
    train_test_split(xs, ys, test_size=0.4, random_state=42)


class NN:
    def __init__(self):
        self.x = create_zeros(2, 1)
        self.weights_1 = create_randoms(3, 2, 42)
        self.bias_1 = create_randoms(3, 1, 42)
        self.weights_2 = create_randoms(3, 3, 42)
        self.bias_2 = create_randoms(3, 1, 42)
        self.weights_3 = create_randoms(3, 3, 42)
        self.bias_3 = create_randoms(3, 1, 42)

        tmp = mat_mul(self.weights_1, self.x)
        tmp = mat_add(self.bias_1, tmp)
        tmp = [[ReLU(i) for i in row] for row in tmp]
        tmp = mat_mul(self.weights_2, tmp)
        tmp = mat_add(self.bias_2, tmp)
        tmp = [[ReLU(i) for i in row] for row in tmp]
        tmp = mat_mul(self.weights_3, tmp)
        self.y_pred = mat_add(self.bias_3, tmp)

        self.y_true = create_zeros(3, 1)

        self.params = []
        self.params.append(self.weights_1)
        self.params.append(self.bias_1)
        self.params.append(self.weights_2)
        self.params.append(self.bias_2)
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
for i in range(50):
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

    print(f'[{i}] Accuracy: {acc}, Best: {best_acc}')

    if best_acc == 1:
        break

end = time.time()
print(f'Time taken: {end - start}')

plot_decision_boundary(nn, X_test, y_test, './images/alpha-boundary.png')
