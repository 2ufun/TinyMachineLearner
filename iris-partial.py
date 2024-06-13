import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from TinyLearner import *
from helper import plot_decision_boundary, softmax, softmax_for_network

# %% read data
df = pd.read_csv('data/iris.data', header=None)
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

acc_record = []
y_logits = []
for ax in X_test:
    y_logits.append(classifier(ax))
y_pred = np.array([softmax(ay).argmax() for ay in y_logits])

acc = 100 * (np.sum(y_pred == y_test) / len(y_test))
acc_record.append(acc)

best_acc = 0
epochs = 100
for epoch in range(1, epochs + 1):
    # train loop
    for ax, ay in zip(X_train, y_train):
        classifier.fit(ax, ay)

    # test loop
    train_y_logits = []
    for ax in X_train:
        train_y_logits.append(classifier(ax))
    y_pred_train = np.array([softmax(ay).argmax() for ay in train_y_logits])

    train_acc = 100 * (np.sum(y_pred_train == y_train) / len(y_train))

    # test loop
    y_logits = []
    for ax in X_test:
        y_logits.append(classifier(ax))
    y_pred = np.array([softmax(ay).argmax() for ay in y_logits])

    acc = 100 * (np.sum(y_pred == y_test) / len(y_test))
    acc_record.append(acc)

    if acc > best_acc:
        best_acc = acc
        plot_decision_boundary(classifier, X_test, y_test, 'images/sgd_boundary.png')

    print(f'[{epoch}] Accuracy: {acc:.2f}%, Train Acc: {train_acc:.2f}%, '
          f'Best Accuracy: {best_acc:.2f}%')

idx = np.argmax(acc_record)
max_v = np.max(acc_record)
plt.plot(acc_record)
plt.xlabel('Epoch')
plt.ylabel('Accuracy(%)')
plt.grid(True)
plt.scatter(idx, max_v, marker='x', color='r', s=50, label='Max value')
plt.text(idx, max_v, f'epoch:{idx}\naccuracy:{max_v:.2f}%',
         color='black', fontsize=14, ha='center', va='bottom')
plt.savefig(f'images/sgd_accuracy_figure.png')
plt.show()
