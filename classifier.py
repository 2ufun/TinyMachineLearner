import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from TinyLearner import *

# %% read data
df = pd.read_csv('data/iris.data', header=None)
X = np.array(df.iloc[:, 2:4])
name_dict = {'Iris-setosa': 0, 'Iris-versicolor': 1}
y = df.iloc[:, 4].map(name_dict).to_numpy()

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.4, random_state=42)


# %% build neural network


class Classifier:
    def __init__(self, optimum):
        self.x = [[Number(0)], [Number(0)]]
        self.weights_from_input = create_randoms(5, 2, seed=42)
        self.bias_from_input = create_randoms(5, 1, seed=42)
        self.weights_to_output = create_randoms(1, 5, seed=42)
        self.bias_to_output = create_randoms(1, 1, seed=42)

        tmp = mat_mul(self.weights_from_input, self.x)
        tmp = mat_add(self.bias_from_input, tmp)
        tmp = [[ReLU(i) for i in row] for row in tmp]
        tmp = mat_mul(self.weights_to_output, tmp)
        self.net = mat_add(self.bias_to_output, tmp)

        self.y = Number(0)
        d = Sub(self.y, self.net[0][0])
        self.loss = Abs(d)

        self.optimum = optimum
        self.optimum.set_params([self.weights_from_input,
                                 self.bias_from_input,
                                 self.weights_to_output,
                                 self.bias_to_output])

    def __call__(self, x) -> float:
        self.x[0][0].set_value(x[0])
        self.x[1][0].set_value(x[1])
        return self.net[0][0].value()

    def fit(self, x, y):
        self.x[0][0].set_value(x[0])
        self.x[1][0].set_value(x[1])
        self.y.set_value(y)
        self.optimum.step(self.loss)


classifier = Classifier(Adam(lr=0.01))


# %% train model


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def save_state():
    global y_logits, y_pred, acc
    y_logits = np.array([classifier(x) for x in X])
    mse = np.mean(np.square(y - y_logits))
    y_pred = [np.round(sigmoid(y_logits))]
    acc = np.sum(y_pred == y) / len(y)
    plt.title(f'MSE={mse:.6f}')
    plt.scatter(x=X[:, 0], y=X[:, 1], c=y_pred, cmap=plt.cm.RdYlBu)
    plt.savefig('images/best.png')


best_acc = 0
acc_record = []
mse_record = []
for epoch in range(100):
    for ax, ay in zip(X_train, y_train):
        classifier.fit(ax, ay)

    y_logits = np.array([classifier(x) for x in X_test])
    mse = np.mean(np.square(y_test - y_logits))
    y_pred = [np.round(sigmoid(y_logits))]
    acc = np.sum(y_pred == y_test) / len(y_test)

    acc_record.append(acc)
    mse_record.append(mse)

    if acc > best_acc:
        best_acc = acc
        save_state()

    if epoch % 10 == 0:
        print(f'[{epoch}] Accuracy: {acc}, Best Accuracy: {best_acc}')

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Accuracy Figure')
plt.plot(acc_record)
plt.subplot(1, 2, 2)
plt.title('MSE Figure')
plt.plot(mse_record)
plt.savefig('images/record.png')
