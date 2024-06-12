import math


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


class Number(Expression):
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

    def grad(self, x: Number) -> float:
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


class Pow(Expression):
    def __init__(self, a: Expression, b: Expression):
        self.a = a
        self.b = b

    def value(self) -> float:
        return pow(self.a.value(), self.b.value())

    def grad(self, x) -> float:
        res = self.b.value() * pow(self.a.value(), self.b.value() - 1)
        if self.a.has_symbol(x):
            res *= self.a.grad(x)
        if self.b.has_symbol(x):
            res *= self.b.grad(x)
        return res

    def has_symbol(self, x) -> bool:
        return self.a.has_symbol(x) or self.b.has_symbol(x)


class Exp(Expression):
    def __init__(self, a: Expression):
        self.a = a

    def value(self) -> float:
        return math.exp(self.a.value())

    def grad(self, x) -> float:
        res = math.exp(self.a.value())
        if self.a.has_symbol(x):
            res *= self.a.grad(x)
        return res

    def has_symbol(self, x) -> bool:
        return self.a.has_symbol(x)


class Tanh(Expression):
    def __init__(self, a: Expression):
        self.exp = Div(Sub(Exp(Mul(Const(2), a)), Exp(a)),
                       Add(Exp(Mul(Const(2), a)), Exp(a)))

    def value(self) -> float:
        return self.exp.value()

    def grad(self, x) -> float:
        return self.exp.grad(x)

    def has_symbol(self, x) -> bool:
        return self.exp.has_symbol(x)


class Sigmoid(Expression):
    def __init__(self, a: Expression):
        self.exp = Div(Exp(a), Add(Number(1), Exp(a)))

    def value(self) -> float:
        return self.exp.value()

    def grad(self, x) -> float:
        return self.exp.grad(x)

    def has_symbol(self, x) -> bool:
        return self.exp.has_symbol(x)
