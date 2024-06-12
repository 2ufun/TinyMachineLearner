import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

from TinyLearner import *

# %% read data
df = pd.read_csv('data/mlp_demo_data_part1.csv')

xs = np.array(df[['u(k)', 'y(k)']])
ys = np.array(df['y1'])

X_train, X_test, y_train, y_test = \
    train_test_split(xs, ys, test_size=0.4, random_state=42)


# %% build neural network
class NN:
    def __init__(self, optimum):
        self.x = [[Number(0)], [Number(0)]]
        self.weights_from_input = create_randoms(3, 2, seed=42)
        self.weights_to_output = create_randoms(1, 3, seed=42)

        tmp = mat_mul(self.weights_from_input, self.x)
        tmp = [[ReLU(i) for i in row] for row in tmp]
        self.net = mat_mul(self.weights_to_output, tmp)

        self.y = Number(0)
        d = Sub(self.y, self.net[0][0])
        self.loss = Mul(Const(0.5), Sqr(d))

        self.optimum = optimum
        self.optimum.set_params([self.weights_from_input, self.weights_to_output])

    def __call__(self, x) -> float:
        self.x[0][0].set_value(x[0])
        self.x[1][0].set_value(x[1])
        return self.net[0][0].value()

    def fit(self, x, y):
        self.x[0][0].set_value(x[0])
        self.x[1][0].set_value(x[1])
        self.y.set_value(y)
        self.optimum.step(self.loss)


nn = NN(Adam(lr=0.01))

# %% train and test model
for _ in range(2):
    for ax, ay in zip(X_train, y_train):
        nn.fit(ax, ay)

y_preds = [nn(x) for x in xs]

plt.plot(y_preds)
plt.plot(ys)
plt.grid(True)
plt.legend(['y_preds', 'y'])
plt.show()
