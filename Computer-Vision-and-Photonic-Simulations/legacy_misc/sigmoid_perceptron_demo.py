# --- LEGACY/MISC FILE ---
# This script does not belong to any module in the target repository
# architecture (Computer Vision / Photonic Simulations / Deep Learning).
# It is kept here unlinked, for reference only, and is not imported by
# anything in src/ or apps/. See legacy_misc/README.md for details.
# ------------------------

import numpy as np

# Define sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Initialize parameters
w1 = 0.5
w2 = -0.5
w3 = 0.3
w4 = -0.4
w5 = 0.2
b = 0.1
c = -0.1

# Input data
x1 = 0.7
x2 = -0.3

# Forward propagation
h2 = w3 * x1 + w1 * x2
h1 = sigmoid(h2)
y_pred = sigmoid(w4 * h1 + w5 * h2 + c)

# True label
y_true = 1

# Calculate cost (cross-entropy)
cost = -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)

# Backpropagation
dL_dy = (y_pred - y_true)
dL_dw4 = dL_dy * h1
dL_dw5 = dL_dy * h2
dL_dc = dL_dy
dL_dh1 = dL_dy * w4
dL_dh2 = dL_dy * w5
dL_dw1 = dL_dh1 * h1 * (1 - h1) * x2
dL_dw3 = dL_dh2 * h2 * (1 - h2) * x1

# Update parameters using learning rate (alpha)
alpha = 0.1
w1 -= alpha * dL_dw1
w3 -= alpha * dL_dw3
w4 -= alpha * dL_dw4
w5 -= alpha * dL_dw5
c -= alpha * dL_dc

print("Updated weights and bias:")
print("w1:", w1)
print("w3:", w3)
print("w4:", w4)
print("w5:", w5)
print("c:", c)
