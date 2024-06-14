import numpy as np


class Expression:
    def value(self) -> float:
        raise NotImplementedError

    def grad_backward(self, grad_value=1.0) -> None:
        raise NotImplementedError


class Const(Expression):
    def __init__(self, v):
        self.v = v

    def value(self) -> float:
        return self.v

    def grad_backward(self, grad_value=1.0) -> None:
        pass


class Var(Expression):
    def __init__(self, v, require_grad=False):
        self.v = v
        self.g = 0
        self.require_grad = require_grad

    def value(self) -> float:
        return self.v

    def grad_backward(self, grad_value=1.0) -> None:
        if self.require_grad:
            self.g += grad_value

    def zero_grad(self) -> None:
        self.g = 0


class Add(Expression):
    def __init__(self, *exps):
        self.exps = exps

    def value(self) -> float:
        return sum(map(lambda x: x.value(), self.exps))

    def grad_backward(self, grad_value=1.0) -> None:
        for exp in self.exps:
            exp.grad_backward(grad_value)


class Sub(Expression):
    def __init__(self, a: Expression, b: Expression):
        self.a = a
        self.b = b

    def value(self) -> float:
        return self.a.value() - self.b.value()

    def grad_backward(self, grad_value=1.0) -> None:
        self.a.grad_backward(grad_value)
        self.b.grad_backward(-grad_value)


class Neg(Expression):
    def __init__(self, v):
        self.v = v

    def value(self) -> float:
        return -self.v.value()

    def grad_backward(self, grad_value=1.0) -> None:
        self.v.grad_backward(-grad_value)


class Mul(Expression):
    def __init__(self, a: Expression, b: Expression):
        self.a = a
        self.b = b

    def value(self) -> float:
        return self.a.value() * self.b.value()

    def grad_backward(self, grad_value=1.0) -> None:
        self.a.grad_backward(self.b.value() * grad_value)
        self.b.grad_backward(self.a.value() * grad_value)


class Div(Expression):
    def __init__(self, a: Expression, b: Expression):
        self.a = a
        self.b = b

    def value(self) -> float:
        return self.a.value() / self.b.value()

    def grad_backward(self, grad_value=1.0) -> None:
        self.a.grad_backward(1 / self.b.value() * grad_value)
        tmp = -(self.a.value() / (self.b.value() ** 2))
        self.b.grad_backward(tmp * grad_value)


class Sqr(Expression):
    def __init__(self, a: Expression):
        self.a = a

    def value(self) -> float:
        return self.a.value() ** 2

    def grad_backward(self, grad_value=1.0) -> None:
        self.a.grad_backward(2 * self.a.value() * grad_value)


class Abs(Expression):
    def __init__(self, a: Expression):
        self.a = a

    def value(self) -> float:
        return abs(self.a.value())

    def grad_backward(self, grad_value=1.0) -> None:
        if self.a.value() >= 0:
            self.a.grad_backward(grad_value)
        else:
            self.a.grad_backward(-grad_value)


class Exp(Expression):
    def __init__(self, a: Expression):
        self.a = a

    def value(self) -> float:
        return np.exp(self.a.value())

    def grad_backward(self, grad_value=1.0) -> None:
        self.a.grad_backward(np.exp(self.a.value()) * grad_value)


class Max(Expression):
    def __init__(self, a: Expression, b: Expression):
        self.a = a
        self.b = b

    def value(self) -> float:
        return max(self.a.value(), self.b.value())

    def grad_backward(self, grad_value=1.0) -> None:
        if self.a.value() > self.b.value():
            self.a.grad_backward(grad_value)
            self.b.grad_backward(0)
        elif self.b.value() > self.a.value():
            self.a.grad_backward(0)
            self.b.grad_backward(grad_value)
        else:
            self.a.grad_backward(grad_value / 2)
            self.b.grad_backward(grad_value / 2)


class Log(Expression):
    def __init__(self, a: Expression):
        self.a = a

    def value(self) -> float:
        return np.log(self.a.value())

    def grad_backward(self, grad_value=1.0) -> None:
        self.a.grad_backward(1 / self.a.value() * grad_value)


class Tanh(Expression):
    def __init__(self, a: Expression):
        self.exp = Div(Sub(Exp(Mul(Const(2), a)), Const(1)),
                       Add(Exp(Mul(Const(2), a)), Const(1)))

    def value(self) -> float:
        return self.exp.value()

    def grad_backward(self, grad_value=1.0) -> None:
        self.exp.grad_backward(grad_value)


class ReLU(Expression):
    def __init__(self, a: Expression):
        self.exp = Max(Const(0), a)

    def value(self) -> float:
        return self.exp.value()

    def grad_backward(self, grad_value=1.0) -> None:
        self.exp.grad_backward(grad_value)
