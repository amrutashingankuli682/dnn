# 2(a) Design a single unit perceptron for classification of a linearly separable binary dataset without using pre-defined models.Use the perceptron() from sklearn .
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import Perceptron

# Generate a linearly separable binary dataset
x, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42)

# Train a Perceptron model
perceptron = Perceptron(max_iter=1000)
perceptron.fit(x, y)

# Plot the decision boundary
x_min, x_max = x[:, 0].min(), x[:, 0].max()
y_min, y_max = x[:, 1].min(), x[:, 1].max()
xx, yy = np.meshgrid(np.linspace(x_min, x_max), np.linspace(y_min, y_max))
Z = perceptron.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# Plot the decision boundary and data points
plt.contourf(xx, yy,Z, alpha=0.8)
plt.scatter(x[:, 0], x[:, 1], c=y, s=60, edgecolor='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Perceptron Decision Boundary')
plt.show()