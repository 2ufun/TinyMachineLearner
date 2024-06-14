import random

from TinyGraderAlpha import *


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
