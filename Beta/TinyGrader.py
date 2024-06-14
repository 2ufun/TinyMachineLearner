import numpy as np


class Expression:
    def value(self) -> float:
        raise NotImplementedError

    def grad(self, x) -> float:
        raise NotImplementedError

    def has_symbol(self, x) -> bool:
        raise NotImplementedError


class Const(Expression):
    def __init__(self, v):
        self.v = v

    def value(self) -> float:
        return self.v

    def grad(self, x) -> float:
        return 0

    def has_symbol(self, x) -> bool:
        return False


class Var(Expression):
    def __init__(self, v: float):
        self.v = v

    def value(self) -> float:
        return self.v

    def set_value(self, v: float) -> None:
        self.v = v

    def grad(self, x) -> float:
        if x == self:
            return 1
        else:
            return 0

    def has_symbol(self, x) -> bool:
        return self == x


class Add(Expression):
    def __init__(self, *args):
        self.args = args

    def value(self) -> float:
        return sum(map(lambda x: x.value(), self.args))

    def grad(self, x) -> float:
        return sum(map(lambda a: a.grad(x), self.args))

    def has_symbol(self, x) -> bool:
        for arg in self.args:
            if arg.has_symbol(x):
                return True
        return False


class Sub(Expression):
    def __init__(self, a: Expression, b: Expression):
        self.a = a
        self.b = b

    def value(self) -> float:
        return self.a.value() - self.b.value()

    def grad(self, x) -> float:
        return self.a.grad(x) - self.b.grad(x)

    def has_symbol(self, x) -> bool:
        return self.a.has_symbol(x) or self.b.has_symbol(x)


class Neg(Expression):
    def __init__(self, v):
        self.v = v

    def value(self) -> float:
        return -self.v.value()

    def grad(self, x) -> float:
        return -self.v.grad(x)

    def has_symbol(self, x) -> bool:
        return self.v.has_symbol(x)


class Mul(Expression):
    def __init__(self, a: Expression, b: Expression):
        self.a = a
        self.b = b

    def value(self) -> float:
        return self.a.value() * self.b.value()

    def grad(self, x) -> float:
        return self.a.grad(x) * self.b.value() + \
            self.a.value() * self.b.grad(x)

    def has_symbol(self, x) -> bool:
        return self.a.has_symbol(x) or self.b.has_symbol(x)


class Div(Expression):
    def __init__(self, a: Expression, b: Expression):
        self.a = a
        self.b = b

    def value(self) -> float:
        return self.a.value() / self.b.value()

    def grad(self, x: Var) -> float:
        x = self.a.grad(x) * self.b.value() - \
            self.a.value() * self.b.grad(x)
        y = self.b.value() ** 2
        return x / y

    def has_symbol(self, x) -> bool:
        return self.a.has_symbol(x) or self.b.has_symbol(x)


class Abs(Expression):
    def __init__(self, a: Expression):
        self.a = a

    def value(self) -> float:
        return abs(self.a.value())

    def grad(self, x) -> float:
        if self.a.value() > 0:
            return self.a.grad(x)
        else:
            return -self.a.grad(x)

    def has_symbol(self, x) -> bool:
        return self.a.has_symbol(x)


class Sqr(Expression):
    def __init__(self, a: Expression):
        self.exp = Mul(a, a)

    def value(self) -> float:
        return self.exp.value()

    def grad(self, x) -> float:
        return self.exp.grad(x)

    def has_symbol(self, x) -> bool:
        return self.exp.has_symbol(x)


class Exp(Expression):
    def __init__(self, a: Expression):
        self.a = a

    def value(self) -> float:
        return np.exp(self.a.value())

    def grad(self, x) -> float:
        if self.a.has_symbol(x):
            res = np.exp(self.a.value())
            res *= self.a.grad(x)
            return res
        else:
            return 0

    def has_symbol(self, x) -> bool:
        return self.a.has_symbol(x)


class Log(Expression):
    def __init__(self, a: Expression):
        self.a = a

    def value(self) -> float:
        return np.log(self.a.value())

    def grad(self, x) -> float:
        if self.a.has_symbol(x):
            res = 1 / self.a.value()
            res *= self.a.grad(x)
            return res
        else:
            return 0

    def has_symbol(self, x) -> bool:
        return self.a.has_symbol(x)


class Max(Expression):
    def __init__(self, a: Expression, b: Expression):
        self.a = a
        self.b = b

    def value(self) -> float:
        return max(self.a.value(), self.b.value())

    def grad(self, x) -> float:
        if self.a.value() > self.b.value():
            return self.a.grad(x)
        elif self.b.value() > self.a.value():
            return self.b.grad(x)
        else:
            return (self.a.grad(x) + self.b.grad(x)) / 2

    def has_symbol(self, x) -> bool:
        return self.a.has_symbol(x) or self.b.has_symbol(x)


# tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
class Tanh(Expression):
    def __init__(self, a: Expression):
        self.exp = Div(Sub(Exp(Mul(Const(2), a)), Const(1)),
                       Add(Exp(Mul(Const(2), a)), Const(1)))

    def value(self) -> float:
        return self.exp.value()

    def grad(self, x) -> float:
        return self.exp.grad(x)

    def has_symbol(self, x) -> bool:
        return self.exp.has_symbol(x)


# sigmoid(x) = 1 / (1 + exp(-x))
class Sigmoid(Expression):
    def __init__(self, a: Expression):
        self.exp = Div(Exp(a), Add(Const(1), Exp(a)))

    def value(self) -> float:
        return self.exp.value()

    def grad(self, x) -> float:
        return self.exp.grad(x)

    def has_symbol(self, x) -> bool:
        return self.exp.has_symbol(x)


# ReLU(x) = max(x, 0)
class ReLU(Expression):
    def __init__(self, a: Expression):
        self.exp = Max(Const(0), a)

    def value(self) -> float:
        return self.exp.value()

    def grad(self, x) -> float:
        return self.exp.grad(x)

    def has_symbol(self, x) -> bool:
        return self.exp.has_symbol(x)
