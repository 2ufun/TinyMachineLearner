import random

from Alpha.TinyGrader import *


# %% Matrix functions
def create_matrix(rows: int, cols: int, builder) -> list:
    mat = []
    for i in range(rows):
        row = []
        for j in range(cols):
            row.append(builder(i, j))
        mat.append(row)
    return mat


def create_zeros(rows: int, cols: int) -> list:
    return create_matrix(rows, cols, lambda i, j: Var(0, True))


def create_randoms(rows: int, cols: int, seed=None) -> list:
    random.seed(seed)
    return create_matrix(rows, cols, lambda i, j: Var(random.random(), True))


def mat_mul(m1: list, m2: list) -> list:
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


def mat_add(m1: list, m2: list) -> list:
    if len(m1) != len(m2) or len(m1[0]) != len(m2[0]):
        raise Exception

    res = []
    for i in range(len(m1)):
        row = []
        for j in range(len(m1[0])):
            row.append(Add(m1[i][j], m2[i][j]))
        res.append(row)

    return res


# %% Optimizers
class SGD:
    def __init__(self, lr=0.01):
        self.params = None
        self.lr = lr

    def set_params(self, params):
        self.params = params

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
    def __init__(self, lr=0.01, mu=0.9):
        self.params = None

        self.lr = lr
        self.mu = mu

    def set_params(self, params):
        self.params = [[[[w, 0] for w in ws] for ws in _] for _ in params]

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
                    delta = self.mu * w[1] - self.lr * w[0].g
                    w[0].v = w[0].v + delta
                    w[1] = delta


eps = 1
while 1 < 1 + eps:
    eps /= 2
eps *= 2


def smooth(beta, average, g):
    return beta * average + (1 - beta) * g


class RMSGD:
    def __init__(self, lr=0.01, beta=0.9):
        self.params = None
        self.lr = lr
        self.beta = beta

    def set_params(self, params):
        self.params = [[[[w, 0] for w in ws] for ws in _] for _ in params]

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
                    w[0].v = w[0].v - lr * w[0].g
                    w[1] = r


class Adam:
    def __init__(self, lr=0.01, beta=0.9, mu=0.9):
        self.params = None

        self.lr = lr
        self.mu = mu
        self.beta = beta

    def set_params(self, params):
        self.params = [[[[w, 0, 0] for w in ws] for ws in _] for _ in params]

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

