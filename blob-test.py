from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

from tiny_machine_learner.tiny_grader import *
from tiny_machine_learner.tiny_learner import *
from tiny_machine_learner.helper import *

X, y = make_blobs(
    n_samples=500, n_features=2, centers=4, cluster_std=1.5, random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42
)


class NN:
    def __init__(self):
        self.x = create_zeros(2, 1)
        self.weights_1 = create_randoms(20, 2, 42)
        self.bias_1 = create_randoms(20, 1, 42)
        self.weights_2 = create_randoms(4, 20, 42)
        self.bias_2 = create_randoms(4, 1, 42)

        tmp = mat_mul(self.weights_1, self.x)
        tmp = mat_add(self.bias_1, tmp)
        tmp = [[ReLU(i) for i in row] for row in tmp]
        tmp = mat_mul(self.weights_2, tmp)
        self.y_pred = mat_add(self.bias_2, tmp)

        self.y_true = create_zeros(4, 1)

        self.params = []
        self.params.append(self.weights_1)
        self.params.append(self.bias_1)
        self.params.append(self.weights_2)
        self.params.append(self.bias_2)

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
for i in range(100):
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
        plot_decision_boundary(nn, X_test, y_test, "images/blob-boundary.png")

    print(f"[{i}] Accuracy: {acc:.5f}, Best: {best_acc:.5f}")

    if best_acc == 1:
        break
