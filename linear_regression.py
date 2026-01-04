import numpy as np
import matplotlib.pyplot as plt

x = np.array([])
y = np.array([])

x_mean, y_mean = x.mean(), y.mean()
slope = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
intercept = y_mean - slope * x_mean
y_pred = slope * x + intercept
r_squared = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - y_mean) ** 2)

plt.figure(figsize=(8, 6))
plt.scatter(x, y, label='Data')
plt.plot(x, y_pred, color='red', label=f'y = {slope:.2f}x + {intercept:.2f}')
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'Linear Regression (R² = {r_squared:.3f})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('regression.png', dpi=150)
