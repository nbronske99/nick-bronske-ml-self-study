import numpy as np
import matplotlib.pyplot as plt

# Newton's method finds where f(x) = 0 by repeatedly following the tangent line
# Formula: x_new = x - f(x) / f'(x)

# --- Root finding: solve x^2 - 2 = 0 to find sqrt(2) ---
f = lambda x: x**2 - 2
f_prime = lambda x: 2*x

x = 1.0
history = [x]

for i in range(10):
    x = x - f(x) / f_prime(x)
    history.append(x)
    if abs(f(x)) < 1e-12:
        break

print(f"Found sqrt(2) = {x:.15f} in {len(history)-1} steps")
print(f"Actual sqrt(2) = {np.sqrt(2):.15f}")

# --- Optimization: minimize f(x) = x^4 - 3x^3 + 2 ---
# Minimum of f(x) is where f'(x) = 0, so apply Newton's method to f'(x)
# x_new = x - f'(x) / f''(x)
# Compare to gradient descent: x_new = x - lr * f'(x)
# Newton replaces the fixed learning rate with 1/f''(x) — an adaptive step size

g = lambda x: x**4 - 3*x**3 + 2
g_prime = lambda x: 4*x**3 - 9*x**2
g_double_prime = lambda x: 12*x**2 - 18*x

# Newton's method
x_n = 3.0
newton_hist = [x_n]
for i in range(100):
    x_n = x_n - g_prime(x_n) / g_double_prime(x_n)
    newton_hist.append(x_n)
    if abs(g_prime(x_n)) < 1e-10:
        break

# Gradient descent
x_g = 3.0
gd_hist = [x_g]
lr = 0.01
for i in range(500):
    x_g = x_g - lr * g_prime(x_g)
    gd_hist.append(x_g)
    if abs(g_prime(x_g)) < 1e-10:
        break

print(f"\nNewton: {len(newton_hist)-1} steps, found minimum at x = {newton_hist[-1]:.6f}")
print(f"Gradient descent: {len(gd_hist)-1} steps, found minimum at x = {gd_hist[-1]:.6f}")

# --- Plot ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left: root finding tangent lines
x_range = np.linspace(0.5, 2.5, 200)
ax1.plot(x_range, f(x_range), 'b-', linewidth=2)
ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
for i in range(min(3, len(history)-1)):
    xi = history[i]
    slope = f_prime(xi)
    x_next = history[i+1]
    ax1.plot([xi, x_next], [f(xi), 0], 'r--', alpha=0.6)
    ax1.plot(xi, f(xi), 'ro', markersize=8)
ax1.plot(history[-1], 0, 'g*', markersize=15)
ax1.set_title('Root Finding: tangent lines converge to √2')
ax1.grid(True, alpha=0.3)

# Right: convergence comparison
newton_err = [abs(g_prime(x)) for x in newton_hist]
gd_err = [abs(g_prime(x)) for x in gd_hist]
ax2.semilogy(newton_err, 'r-o', label="Newton's method", markersize=6)
ax2.semilogy(gd_err, 'b-o', label='Gradient descent', markersize=3)
ax2.set_xlabel('Iteration')
ax2.set_ylabel('|f\'(x)|')
ax2.set_title('Convergence: Newton vs Gradient Descent')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('newtons_method.png', dpi=150)
