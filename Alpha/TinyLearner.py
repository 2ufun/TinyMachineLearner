import random

from Alpha.TinyGrader import *


# %% Matrix functions
def create_matrix(rows: int, cols: int, builder) -> list:
    return [[builder(i, j) for j in range(cols)] for i in range(rows)]


def create_zeros(rows: int, cols: int) -> list[list[Var]]:
    return create_matrix(rows, cols, lambda i, j: Var(0, True))


def create_randoms(rows: int, cols: int, seed=None) -> list[list[Var]]:
    random.seed(seed)
    return create_matrix(rows, cols, lambda i, j: Var(random.random(), True))


def mat_mul(m1: list, m2: list) -> list[list[Expression]]:
    if len(m1[0]) != len(m2):
        raise Exception

    m = []
    for i in range(len(m1)):
        row = []
        for j in range(len(m2[0])):
            pairs = []
            for k in range(len(m2)):
                pairs.append(Mul(m1[i][k], m2[k][j]))
            row.append(Add(*pairs))
        m.append(row)

    return m


def mat_add(m1: list, m2: list) -> list[list[Expression]]:
    if len(m1) != len(m2) or len(m1[0]) != len(m2[0]):
        raise Exception

    return [[Add(m1[i][j], m2[i][j])
             for j in range(len(m1[0]))]
            for i in range(len(m1))]


# %% Loss Functions
def structural_loss(exps: list) -> list:
    tmp = [Exp(exp) for exp in exps]
    n = Add(*tmp)
    return [Div(Exp(exp), n) for exp in exps]


class CrossEntropyLoss:
    def __init__(self, model):
        prob = structural_loss([n[0] for n in model.y_pred])
        prod = [Mul(model.y_true[i][0], Log(prob[i])) for i in range(len(model.y_true))]
        self.loss = Neg(Add(*prod))

    def backward(self):
        self.loss.grad_backward()


# %% Optimizers
class SGD:
    def __init__(self, params, lr=0.01):
        self.params = params
        self.lr = lr

    def zero_grad(self):
        for layer in self.params:
            for ws in layer:
                for w in ws:
                    w.zero_grad()

    def step(self):
        for layer in self.params:
            for ws in layer:
                for w in ws:
                    w.v = w.v - self.lr * w.g


class MGD:
    def __init__(self, params, lr=0.01, mu=0.9):
        self.params = [[[[w, 0] for w in ws] for ws in _] for _ in params]

        self.lr = lr
        self.mu = mu

    def zero_grad(self):
        for layer in self.params:
            for ws in layer:
                for w in ws:
                    w[0].zero_grad()

    def step(self):
        for layer in self.params:
            for ws in layer:
                for w in ws:
                    delta = self.mu * w[1] - self.lr * w[0].g
                    w[0].v = w[0].v + delta
                    w[1] = delta


def smooth(beta, average, g):
    return beta * average + (1 - beta) * g


class RMSGD:
    def __init__(self, params, lr=0.01, beta=0.9):
        self.params = [[[[w, 0] for w in ws] for ws in _] for _ in params]
        self.lr = lr
        self.beta = beta

    def zero_grad(self):
        for layer in self.params:
            for ws in layer:
                for w in ws:
                    w[0].zero_grad()

    def step(self):
        for layer in self.params:
            for ws in layer:
                for w in ws:
                    r = smooth(self.beta, w[1], w[0].g ** 2)
                    lr = self.lr / (np.sqrt(r) + eps)
                    w[0].v = w[0].v - lr * w[0].g
                    w[1] = r


class Adam:
    def __init__(self, params, lr=0.01, beta=0.9, mu=0.9):
        self.params = [[[[w, 0, 0] for w in ws] for ws in _] for _ in params]

        self.lr = lr
        self.mu = mu
        self.beta = beta

    def zero_grad(self):
        for layer in self.params:
            for ws in layer:
                for w in ws:
                    w[0].zero_grad()

    def step(self):
        if self.params is None:
            raise Exception

        for layer in self.params:
            for ws in layer:
                for w in ws:
                    r = smooth(self.beta, w[1], w[0].g ** 2)
                    lr = self.lr / (np.sqrt(r) + eps)
                    v = smooth(self.mu, w[2], w[0].g)
                    delta = lr * v
                    w[0].v = w[0].v - delta
                    w[1] = r
                    w[2] = v
