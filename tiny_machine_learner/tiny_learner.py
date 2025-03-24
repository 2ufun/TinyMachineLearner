import random
from typing import Callable
from tiny_machine_learner.tiny_grader import *


# Matrix functions


def create_matrix(
    rows: int, cols: int, builder: Callable[[int, int], Expression]
) -> list:
    return [[builder(i, j) for j in range(cols)] for i in range(rows)]


def create_zeros(rows: int, cols: int) -> list[list[Var]]:
    return create_matrix(rows, cols, lambda i, j: Var(0))


def create_randoms(rows: int, cols: int, seed=None) -> list[list[Var]]:
    random.seed(seed)
    return create_matrix(rows, cols, lambda i, j: Var(random.random()))


def mat_mul(
    m1: list[list[Expression]],
    m2: list[list[Expression]],
) -> list[list[Expression]]:
    if len(m1[0]) != len(m2):
        raise Exception

    return [
        [
            Add(*[Mul(m1[i][k], m2[k][j]) for k in range(len(m2))])
            for j in range(len(m2[0]))
        ]
        for i in range(len(m1))
    ]


def mat_add(
    m1: list[list[Expression]],
    m2: list[list[Expression]],
) -> list[list[Expression]]:
    if len(m1) != len(m2) or len(m1[0]) != len(m2[0]):
        raise Exception

    return [
        [Add(m1[i][j], m2[i][j]) for j in range(len(m1[0]))]
        for i in range(len(m1))
    ]


#  Loss Functions


def structural_loss(
    exps: list[Expression],
) -> list[Expression]:
    tmp = [Exp(exp) for exp in exps]
    n = Add(*tmp)
    return [Div(Exp(exp), n) for exp in exps]


class CrossEntropyLoss:
    __slots__ = ["__loss"]

    def __init__(self, model):
        prob = structural_loss([n[0] for n in model.y_pred])
        prod = [
            Mul(model.y_true[i][0], Log(prob[i]))
            for i in range(len(model.y_true))
        ]
        self.__loss = Neg(Add(*prod))

    def backward(self):
        self.__loss.grad_backward()


# Optimizers


class Optimizer(abc.ABC):
    @abc.abstractmethod
    def zero_grad(self) -> None:
        pass

    @abc.abstractmethod
    def step(self) -> None:
        pass


class SGD(Optimizer):
    __slots__ = ["__params", "__lr"]

    def __init__(self, params: list[list[list[Var]]], lr: float = 0.01):
        self.__params = params
        self.__lr = lr

    def zero_grad(self) -> None:
        for layer in self.__params:
            for ws in layer:
                for w in ws:
                    w.zero_grad()

    def step(self) -> None:
        for layer in self.__params:
            for ws in layer:
                for w in ws:
                    w.v = w.v - self.__lr * w.g


class MGD(Optimizer):
    __slots__ = ["__params", "__lr", "__mu"]

    def __init__(
        self,
        params: list[list[list[Var]]],
        lr: float = 0.01,
        mu: float = 0.9,
    ):
        self.__params = [[[[w, 0] for w in ws] for ws in _] for _ in params]

        self.__lr = lr
        self.__mu = mu

    def zero_grad(self) -> None:
        for layer in self.__params:
            for ws in layer:
                for w in ws:
                    w[0].zero_grad()

    def step(self) -> None:
        for layer in self.__params:
            for ws in layer:
                for w in ws:
                    delta = self.__mu * w[1] - self.__lr * w[0].g
                    w[0].v = w[0].v + delta
                    w[1] = delta


def smooth(beta: float, average: float, g: float) -> float:
    return beta * average + (1 - beta) * g


class RMSGD:
    __slots__ = ["__params", "__lr", "__beta"]

    def __init__(
        self,
        params: list[list[list[Var]]],
        lr: float = 0.01,
        beta: float = 0.9,
    ):
        self.__params = [[[[w, 0] for w in ws] for ws in _] for _ in params]
        self.__lr = lr
        self.__beta = beta

    def zero_grad(self) -> None:
        for layer in self.__params:
            for ws in layer:
                for w in ws:
                    w[0].zero_grad()

    def step(self) -> None:
        for layer in self.__params:
            for ws in layer:
                for w in ws:
                    r = smooth(self.__beta, w[1], w[0].g ** 2)
                    lr = self.__lr / (np.sqrt(r) + np.finfo(np.float64).eps)
                    w[0].v = w[0].v - lr * w[0].g
                    w[1] = r


class Adam:
    __slots__ = ["__params", "__lr", "__beta", "__mu"]

    def __init__(
        self,
        params: list[list[list[Var]]],
        lr: float = 0.01,
        beta: float = 0.9,
        mu: float = 0.9,
    ):
        self.__params = [[[[w, 0, 0] for w in ws] for ws in _] for _ in params]

        self.__lr = lr
        self.__mu = mu
        self.__beta = beta

    def zero_grad(self) -> None:
        for layer in self.__params:
            for ws in layer:
                for w in ws:
                    w[0].zero_grad()

    def step(self) -> None:
        for layer in self.__params:
            for ws in layer:
                for w in ws:
                    r = smooth(self.__beta, w[1], w[0].g ** 2)
                    lr = self.__lr / (np.sqrt(r) + np.finfo(np.float64).eps)
                    v = smooth(self.__mu, w[2], w[0].g)
                    delta = lr * v
                    w[0].v = w[0].v - delta
                    w[1] = r
                    w[2] = v
