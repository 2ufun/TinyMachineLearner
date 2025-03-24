import numpy as np
import abc


class Expression(abc.ABC):
    @abc.abstractmethod
    def value(self) -> float:
        pass

    @abc.abstractmethod
    def grad_backward(self, grad_value: float = 1.0) -> None:
        pass


class Const(Expression):
    __slots__ = ["v"]

    def __init__(self, v):
        self.v = v

    def value(self) -> float:
        return self.v

    def grad_backward(self, grad_value: float = 1.0) -> None:
        pass


class Var(Expression):
    __slots__ = ["v", "g"]

    def __init__(self, v: float):
        self.v = v
        self.g = 0

    def value(self) -> float:
        return self.v

    def grad_backward(self, grad_value: float = 1.0) -> None:
        self.g += grad_value

    def zero_grad(self) -> None:
        self.g = 0


class Add(Expression):
    __slots__ = ["__exps"]

    def __init__(self, *exps: Expression):
        self.__exps: tuple[Expression, ...] = exps

    def value(self) -> float:
        return sum((exp.value() for exp in self.__exps))

    def grad_backward(self, grad_value: float = 1.0) -> None:
        for exp in self.__exps:
            exp.grad_backward(grad_value)


class Sub(Expression):
    __slots__ = ["__a", "__b"]

    def __init__(self, a: Expression, b: Expression):
        self.__a = a
        self.__b = b

    def value(self) -> float:
        return self.__a.value() - self.__b.value()

    def grad_backward(self, grad_value: float = 1.0) -> None:
        self.__a.grad_backward(grad_value)
        self.__b.grad_backward(-grad_value)


class Neg(Expression):
    __slots__ = ["__e"]

    def __init__(self, e: Expression):
        self.__e = e

    def value(self) -> float:
        return -self.__e.value()

    def grad_backward(self, grad_value: float = 1.0) -> None:
        self.__e.grad_backward(-grad_value)


class Mul(Expression):
    __slots__ = ["__a", "__b"]

    def __init__(self, a: Expression, b: Expression):
        self.__a = a
        self.__b = b

    def value(self) -> float:
        return self.__a.value() * self.__b.value()

    def grad_backward(self, grad_value: float = 1.0) -> None:
        self.__a.grad_backward(self.__b.value() * grad_value)
        self.__b.grad_backward(self.__a.value() * grad_value)


class Div(Expression):
    __slots__ = ["__a", "__b"]

    def __init__(self, a: Expression, b: Expression):
        self.__a = a
        self.__b = b

    def value(self) -> float:
        b_v = self.__b.value()
        if np.isclose(b_v, 0):
            b_v += np.finfo(np.float64).eps
        return self.__a.value() / b_v

    def grad_backward(self, grad_value: float = 1.0) -> None:
        b_v = self.__b.value()
        if np.isclose(b_v, 0):
            b_v += np.finfo(np.float64).eps
        self.__a.grad_backward(1 / b_v * grad_value)
        self.__b.grad_backward(-(self.__a.value() / (b_v**2)) * grad_value)


class Sqr(Expression):
    __slots__ = ["__e"]

    def __init__(self, e: Expression):
        self.__e = e

    def value(self) -> float:
        return self.__e.value() ** 2

    def grad_backward(self, grad_value: float = 1.0) -> None:
        self.__e.grad_backward(2 * self.__e.value() * grad_value)


class Abs(Expression):
    __slots__ = ["__e"]

    def __init__(self, e: Expression):
        self.__e = e

    def value(self) -> float:
        return abs(self.__e.value())

    def grad_backward(self, grad_value: float = 1.0) -> None:
        if self.__e.value() >= 0:
            self.__e.grad_backward(grad_value)
        else:
            self.__e.grad_backward(-grad_value)


class Exp(Expression):
    __slots__ = ["__e"]

    def __init__(self, e: Expression):
        self.__e = e

    def value(self) -> float:
        return np.exp(self.__e.value())

    def grad_backward(self, grad_value: float = 1.0) -> None:
        self.__e.grad_backward(np.exp(self.__e.value()) * grad_value)


class Max(Expression):
    __slots__ = ["__a", "__b"]

    def __init__(self, a: Expression, b: Expression):
        self.__a = a
        self.__b = b

    def value(self) -> float:
        return max(self.__a.value(), self.__b.value())

    def grad_backward(self, grad_value: float = 1.0) -> None:
        a_v = self.__a.value()
        b_v = self.__b.value()
        if a_v > b_v:
            self.__a.grad_backward(grad_value)
            self.__b.grad_backward(0)
        elif a_v < b_v:
            self.__a.grad_backward(0)
            self.__b.grad_backward(grad_value)
        else:
            self.__a.grad_backward(grad_value / 2)
            self.__b.grad_backward(grad_value / 2)


class Log(Expression):
    __slots__ = ["__e"]

    def __init__(self, e: Expression):
        self.__e = e

    def value(self) -> float:
        return np.log(self.__e.value())

    def grad_backward(self, grad_value: float = 1.0) -> None:
        self.__e.grad_backward(1 / self.__e.value() * grad_value)


class Tanh(Expression):
    __slots__ = ["__exp"]

    def __init__(self, e: Expression):
        self.__exp = Div(
            Sub(Exp(Mul(Const(2), e)), Const(1)),
            Add(Exp(Mul(Const(2), e)), Const(1)),
        )

    def value(self) -> float:
        return self.__exp.value()

    def grad_backward(self, grad_value: float = 1.0) -> None:
        self.__exp.grad_backward(grad_value)


class ReLU(Expression):
    __slots__ = ["__exp"]

    def __init__(self, e: Expression):
        self.__exp = Max(Const(0), e)

    def value(self) -> float:
        return self.__exp.value()

    def grad_backward(self, grad_value: float = 1.0) -> None:
        self.__exp.grad_backward(grad_value)
