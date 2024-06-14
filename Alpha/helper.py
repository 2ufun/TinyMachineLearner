from matplotlib import pyplot as plt

from Alpha.TinyGrader import *


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def softmax_for_network(exps: list) -> list:
    tmp = [Exp(exp) for exp in exps]
    n = Add(*tmp)
    return [Div(Exp(exp), n) for exp in exps]


def plot_decision_boundary(model, X, y, file_name):
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    X_to_pred_on = np.column_stack((xx.ravel(), yy.ravel()))
    y_logits = [model(ax) for ax in X_to_pred_on]
    y_pred = np.array([softmax(ay).argmax() for ay in y_logits])

    y_pred = y_pred.reshape(xx.shape)
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.savefig(file_name)
    plt.show()
