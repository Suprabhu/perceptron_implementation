#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=1000, random_state=1):
        self.learning_rate = learning_rate
        self.epochs = epochs
        np.random.seed(random_state)
        
    def fit(self, X, y):
        # Initialize weights and bias
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        self.errors = []
        
        for _ in range(self.epochs):
            error_count = 0
            for xi, target in zip(X, y):
                update = self.learning_rate * (target - self.predict(xi))
                self.weights += update * xi
                self.bias += update
                error_count += int(update != 0.0)
            self.errors.append(error_count)
    
    def predict(self, X):
        return np.where(np.dot(X, self.weights) + self.bias >= 0.0, 1, -1)

# Load the Iris dataset into a pandas DataFrame
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Prepare the dataset
def prepare_data(df, class_1, class_2, feature_1, feature_2):
    df_filtered = df[df['target'].isin([class_1, class_2])]
    X = df_filtered.iloc[:, [feature_1, feature_2]].values
    y = df_filtered['target'].apply(lambda x: 1 if x == class_1 else -1).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

# Train and evaluate the model with all 4C2 combinations of features
combinations = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
for (f1, f2) in combinations:
    X_prepared, y_prepared = prepare_data(df, 0, 1, f1, f2)
    X_train, X_test, y_train, y_test = train_test_split(X_prepared, y_prepared, test_size=0.2, random_state=1)
    
    perceptron = Perceptron(learning_rate=0.01, epochs=1000, random_state=1)
    perceptron.fit(X_train, y_train)
    
    predictions = perceptron.predict(X_test)
    accuracy = np.mean(predictions == y_test) * 100
    print(f"Features {f1} and {f2} - Accuracy: {accuracy:.2f}%")


# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=1000, random_state=1):
        self.learning_rate = learning_rate
        self.epochs = epochs
        np.random.seed(random_state)
        
    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        self.errors = []
        
        for _ in range(self.epochs):
            error_count = 0
            for xi, target in zip(X, y):
                update = self.learning_rate * (target - self.predict(xi))
                self.weights += update * xi
                self.bias += update
                error_count += int(update != 0.0)
            self.errors.append(error_count)
    
    def predict(self, X):
        return np.where(np.dot(X, self.weights) + self.bias >= 0.0, 1, -1)

# Load the Iris dataset into a pandas DataFrame
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Prepare the dataset for binary classification with two classes and two features
def prepare_data(df, class_1, class_2, feature_1, feature_2):
    df_filtered = df[df['target'].isin([class_1, class_2])]
    X = df_filtered.iloc[:, [feature_1, feature_2]].values
    y = df_filtered['target'].apply(lambda x: 1 if x == class_1 else -1).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

# Visualize decision boundary
def plot_decision_boundary(perceptron, X, y, title="Decision Boundary"):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    Z = perceptron.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, marker='o', edgecolor='k', cmap=plt.cm.coolwarm)
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

# Evaluate and visualize for each feature combination
combinations = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
for (f1, f2) in combinations:
    X_prepared, y_prepared = prepare_data(df, 0, 1, f1, f2)
    X_train, X_test, y_train, y_test = train_test_split(X_prepared, y_prepared, test_size=0.2, random_state=1)
    
    perceptron = Perceptron(learning_rate=0.01, epochs=1000, random_state=1)
    perceptron.fit(X_train, y_train)
    
    # Test accuracy
    predictions = perceptron.predict(X_test)
    accuracy = np.mean(predictions == y_test) * 100
    print(f"Features {f1} and {f2} - Accuracy: {accuracy:.2f}%")
    
    # Plot decision boundary
    plot_decision_boundary(perceptron, X_train, y_train, title=f"Decision Boundary for Features {f1} and {f2}")
    
    # Plot accuracy over epochs
    plt.plot(range(1, len(perceptron.errors) + 1), perceptron.errors, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of Errors')
    plt.title(f'Errors over Epochs for Features {f1} and {f2}')
    plt.show()

