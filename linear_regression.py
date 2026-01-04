import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('diabetes.csv', delimiter=',', skip_header=1)
x = data[:, 5]
y = data[:, 1]

mask = (x > 0) & (y > 0)
x = x[mask]
y = y[mask]

x_mean, y_mean = x.mean(), y.mean()
slope = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
intercept = y_mean - slope * x_mean
y_pred = slope * x + intercept
r_squared = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - y_mean) ** 2)

plt.figure(figsize=(8, 6))
plt.scatter(x, y, alpha=0.5, label='Data')
plt.plot(x, y_pred, color='red', label=f'y = {slope:.2f}x + {intercept:.2f}')
plt.xlabel('BMI')
plt.ylabel('Glucose (mg/dL)')
plt.title(f'Diabetes Dataset: BMI vs Glucose (R² = {r_squared:.3f})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('regression.png', dpi=150)
