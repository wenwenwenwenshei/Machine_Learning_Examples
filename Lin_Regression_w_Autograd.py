import torch
from fastai.learner import *


def lin(a,b,x): return a*x+b

def relu(x): return x * (x > 0)

def gen_fake_data(n, a, b):
    x = s = np.random.uniform(0,1,n) 
    y = lin(a,b,x) + 0.1 * np.random.normal(0,3,n)
    return x, y

x, y = gen_fake_data(50, 3., 8.)

a = V(np.random.randn(1), requires_grad=True) #slope
b = V(np.random.randn(1), requires_grad=True) #intercept

print(f'slope {a.item()} intercept {b.item()} BEFORE')

x, y = V(x), V(y)

def mean_sqt(y_hat, y): return ((y_hat - y) ** 2).mean()

lr = 0.001

def bak_prop(a, b, x):
	loss = (mean_sqt((lin(a,b,x)), y))

	loss.backward()

	a.data -= lr * a.grad.data
	b.data -= lr * b.grad.data

	a.grad.data.zero_()
	b.grad.data.zero_()

	if i % 300 == 0:
		print(f'Loss... {loss.data.item()}')

for i in range(5000):
	bak_prop(a, b, x)

print(f'slope {a.item()} intercept {b.item()} AFTER')




