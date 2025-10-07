import numpy as np


X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 1, 1, 0])


np.random.seed(1)
w1 = np.random.randn(2, 2)
b1 = np.random.randn(2)
w2 = np.random.randn(2)
b2 = np.random.randn()
lr = 0.1
epochs = 5000

def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_deriv(x): return x * (1 - x)

for _ in range(epochs):
    
    z1 = np.dot(X, w1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, w2) + b2
    a2 = sigmoid(z2)

    
    error = y - a2
    d2 = error * sigmoid_deriv(a2)
    d1 = d2.reshape(-1,1) * sigmoid_deriv(a1) * w2

    
    w2 += lr * np.dot(a1.T, d2)
    b2 += lr * np.sum(d2)
    w1 += lr * np.dot(X.T, d1)
    b1 += lr * np.sum(d1, axis=0)


for xi in X:
    a1 = sigmoid(np.dot(xi, w1) + b1)
    a2 = sigmoid(np.dot(a1, w2) + b2)
    print(f"Entrada: {xi}, Salida: {a2.round()}")
