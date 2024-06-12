# Tiny Machine Learner 💻

本人闲来无事，文思泉涌写出来的迷你机器学习库，具有运算速度慢、运行占用内存大、精度低、泛用性差等优点 :P

## Tiny Grader

计算微分的简单办法就是用`(f(x+dx) - f(x)) / dx`来做近似，其中dx的值可以用下面的方法求取：

```python
dx = 1
while 1 < 1 + dx:
    dx /= 2
dx *= 2
```

但我们怎么可能用这种低端的办法，一点都不专业😡（~~说实话实用效果还不错~~），所以我在某一天晚上失眠时写出了这个简陋的自动微分库。

如果只做普通计算，那么就不需要保存计算式的“形状”，而计算微分则要保存这些信息。TinyGrader的核心就是用表达式对象来取代普通的代数式，以最简单的常数和变量为例：

```python
class Const:
    def __init__(self, v):
        self.v = v
    def value(self) -> float:
        return self.v
    def grad(self, x) -> float:
        return 0

class Number:
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
```

这两个就像是数据结构-树的叶子节点一样，是计算式最基本的组成单位，相应的梯度计算不作赘述。然后就是运算符的定义了，以加法为例：

```python
class Add:
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def value(self) -> float:
        return self.a.value() + self.b.value()
    def grad(self, x) -> float:
        return self.a.grad(x) + self.b.grad(x)
```

有了上面三个类，你就可以构造出`x + y + 1`这样的计算式了，计算式的值以及梯度都可以自动求取：

```python
x = Number(0)
y = Number(1)
exp = Add(y, Add(x, Const(1)))
exp.grad(x) # exp在x=0，y=1时对x的梯度
x.set_value(1)
exp.value() # exp在x=1，y=1时的值
```

这样构造出来的表达式就像是数据结构里的二叉树一样，计算梯度和求值实际上就是在做树的遍历，了解过LISP的人可能会感到非常亲切。

类似的我定义了减、乘、除、乘方、e指数等运算符，以及Sigmoid、Tanh这些经典的激活函数，这些东西构成了TinyGrader的所有内容。

## Tiny Learner

这个文件里主要是矩阵有关的函数以及优化算法。

矩阵部分无需多言，一个是创建随机矩阵的函数，另一个是进行矩阵运算的函数。文件里描述的矩阵就是一个二维的Python列表，矩阵乘法也无非是简单的嵌套循环而已。

优化算法里有普通的梯度下降优化器、动量梯度下降优化器、RMSE梯度下降优化器和Adam优化器。后面三个和普通的梯度下降算法不同的地方就是参考了以前的梯度值，以此来调节参数更新的速度。

## Application

测试用例是我们智能控制必修课的一个实验用数据集（~~比玩具还玩具~~）：

![](images/data.png)

我搭建了一个只有单个隐藏层的神经网络，神经元个数只有3个，激活函数用的Tanh，0.01的学习率迭代了20次，拟合效果还不错：

![](images/test.png)

还有经典的石蕊数据集，不过我这个极轻量级的小框架支撑不了多分类问题，所以我删除其中的一个种类，把三分类的数据集转换成了二分类的数据集。为了方便画图，我只用了其中的两个特征（~~你这简化的东西也太多了吧~~），分类效果还不错：

![](images/best.png)

![](images/record.png)

## Future

实际上我不打算再发展这个小玩具了，尽管它bug频多、效率低下（~~我是废物😭~~），可以改善的地方还有很多，但是我因为这个东西已经两天没睡好觉了，我想让我的大脑好好放松一下。

造轮子其实是个有趣的过程，在做这个东西的过程中，我体会到了面向对象的不足、LISP 语法的强大等等东西，这些都是我当脚本小子所感受不到的，如果我的拙作对你有所帮助，那么我很荣幸;)
