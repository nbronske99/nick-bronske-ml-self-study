import numpy as np
import matplotlib.pyplot as plt

# Goal: predict whether a person has diabetes (Outcome = 1) or not (Outcome = 0)
# from their medical measurements, then check the predictions on people the
# model has never seen.

# --- Load data ---
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
           'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
data = np.genfromtxt('diabetes.csv', delimiter=',', skip_header=1)

# A value of 0 for glucose, blood pressure or BMI means "not recorded", so drop those rows.
# SkinThickness and Insulin are left out entirely: about 30% and 49% of their values are missing.
mask = (data[:, 1] > 0) & (data[:, 2] > 0) & (data[:, 5] > 0)
data = data[mask]

features = ['Pregnancies', 'Glucose', 'BloodPressure', 'BMI', 'DiabetesPedigreeFunction', 'Age']
X = data[:, [columns.index(f) for f in features]]
y = data[:, -1]

# --- Train / test split ---
# Hold out 20% of people. The model never sees them during training,
# so its accuracy on them is an honest estimate of how it does on new people.
rng = np.random.default_rng(0)
idx = rng.permutation(len(y))
n_test = len(y) // 5
test_idx, train_idx = idx[:n_test], idx[n_test:]
X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

# Standardize each feature to mean 0, std 1 (using training stats only),
# so a weight's size tells you how much that feature matters.
mu, sigma = X_train.mean(axis=0), X_train.std(axis=0)
standardize = lambda X: (X - mu) / sigma
add_bias = lambda X: np.column_stack([np.ones(len(X)), X])
A_train = add_bias(standardize(X_train))
A_test = add_bias(standardize(X_test))

# --- Model ---
# Linear regression outputs any real number; we need a probability in (0, 1).
# Logistic regression squashes the linear part through the sigmoid:
#   p = sigmoid(w · x) = 1 / (1 + e^(-w · x))
sigmoid = lambda z: 1 / (1 + np.exp(-z))

# Loss (negative log-likelihood / cross-entropy):
#   L(w) = -sum[ y log p + (1 - y) log(1 - p) ]
# Gradient:  A^T (p - y)
# Hessian:   A^T diag(p(1 - p)) A
def loss(w, A, y):
    p = sigmoid(A @ w)
    eps = 1e-12
    return -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))

# --- Train with Newton's method ---
# Same idea as newtons_method.py, now in several dimensions:
#   1D:  x_new = x - f'(x) / f''(x)
#   nD:  w_new = w - H^(-1) ∇L
w = np.zeros(A_train.shape[1])
loss_history = [loss(w, A_train, y_train)]
for i in range(20):
    p = sigmoid(A_train @ w)
    grad = A_train.T @ (p - y_train)
    H = A_train.T @ (A_train * (p * (1 - p))[:, None])
    step = np.linalg.solve(H, grad)
    w = w - step
    loss_history.append(loss(w, A_train, y_train))
    if np.linalg.norm(step) < 1e-10:
        break

print(f"Newton's method converged in {len(loss_history)-1} steps")
print("\nLearned weights (standardized, bigger |w| = more influence):")
for name, wi in sorted(zip(features, w[1:]), key=lambda t: -abs(t[1])):
    print(f"  {name:26s} {wi:+.3f}")

# --- Evaluate on the held-out test set ---
p_test = sigmoid(A_test @ w)
pred = (p_test >= 0.5).astype(float)

accuracy = np.mean(pred == y_test)
baseline = max(y_test.mean(), 1 - y_test.mean())  # always guess the most common answer
tp = np.sum((pred == 1) & (y_test == 1))
tn = np.sum((pred == 0) & (y_test == 0))
fp = np.sum((pred == 1) & (y_test == 0))
fn = np.sum((pred == 0) & (y_test == 1))

print(f"\nTest set: {len(y_test)} people the model never saw")
print(f"  Accuracy:           {accuracy:.1%}")
print(f"  Baseline (guess 0): {baseline:.1%}")
print(f"  Caught {tp:.0f} of {tp + fn:.0f} diabetic cases (recall {tp / (tp + fn):.1%})")
print(f"  {fp:.0f} false alarms, {fn:.0f} missed cases")

# --- Predict for a new person ---
def predict(pregnancies, glucose, blood_pressure, bmi, pedigree, age):
    x = np.array([[pregnancies, glucose, blood_pressure, bmi, pedigree, age]])
    return sigmoid(add_bias(standardize(x)) @ w)[0]

print("\nPredictions for new people:")
print(f"  Age 25, glucose  90, BMI 23: {predict(0, 90, 70, 23, 0.3, 25):.1%} chance of diabetes")
print(f"  Age 50, glucose 160, BMI 35: {predict(3, 160, 80, 35, 0.6, 50):.1%} chance of diabetes")

# --- Plot ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left: training loss per Newton step
ax1.plot(loss_history, 'r-o')
ax1.set_xlabel('Newton step')
ax1.set_ylabel('Training loss (cross-entropy)')
ax1.set_title("Training with Newton's method")
ax1.grid(True, alpha=0.3)

# Right: predicted probability for each test person, split by what actually happened
bins = np.linspace(0, 1, 21)
ax2.hist(p_test[y_test == 0], bins=bins, alpha=0.6, label='No diabetes (actual)')
ax2.hist(p_test[y_test == 1], bins=bins, alpha=0.6, label='Diabetes (actual)')
ax2.axvline(0.5, color='gray', linestyle='--', label='Decision threshold')
ax2.set_xlabel('Predicted probability of diabetes')
ax2.set_ylabel('Number of test people')
ax2.set_title(f'Test set predictions (accuracy {accuracy:.1%})')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('logistic_regression.png', dpi=150)
