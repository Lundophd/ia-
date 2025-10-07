import numpy as np
import matplotlib.pyplot as plt


X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 0, 0, 1])

# Inicialización
np.random.seed(1)
w = np.random.randn(2)
b = np.random.randn()
lr = 0.1
epochs = 30

errors = []


for _ in range(epochs):
    total_error = 0
    for xi, target in zip(X, y):
        z = np.dot(xi, w) + b
        output = z  # salida lineal
        error = target - output
        w += lr * error * xi
        b += lr * error
        total_error += error**2
    errors.append(total_error)

print("Pesos finales:", w)
print("Bias final:", b)


plt.plot(errors)
plt.title("Convergencia del error - Adaline (AND)")
plt.xlabel("Épocas")
plt.ylabel("Error cuadrático")
plt.show()
