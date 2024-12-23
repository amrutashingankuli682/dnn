#  Identify the problem with single unit Perceptron. Classify using Or, And and Xor data and analyze the result
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
# Data for AND, OR, XOR gates
data = {
'AND': (np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([0, 0, 0, 1])),  # Use 0 and 1 for labels
'OR':  (np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([0, 1, 1, 1])),  # Use 0 and 1 for labels
'XOR': (np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([0, 1, 1, 0])),  # Use 0 and 1 for labels
}

def plot_decision_boundary(perceptron, X, Y, gate_name):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    Z = perceptron.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(6, 6))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)
    plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', s=100, cmap=plt.cm.Paired)
    plt.title(f"Decision Boundary for {gate_name} Gate")
    plt.xlabel("Input 1")
    plt.ylabel("Input 2")
    plt.grid(True)
    plt.show()

# Classify AND, OR, XOR gates
for gate, (x, y) in data.items():
   perceptron = Perceptron(max_iter=10, eta0=1, random_state=42)
   perceptron.fit(x, y)
   y_pred = perceptron.predict(x)
   acc = accuracy_score(y, y_pred) * 100
   print(f"{gate} gate accuracy: {acc:.2f}%")
   print(f"Predictions: {y_pred}")
   print(f"True Labels: {y}")
   plot_decision_boundary(perceptron, x, y, gate)