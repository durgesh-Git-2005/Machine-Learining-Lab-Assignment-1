# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Step 1: Load the Dataset
iris = load_iris()
X = iris.data  # features
y = iris.target  # target labels

# Step 2: Data Standardization
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Step 3: Apply LDA
lda = LDA(n_components=2)  # Set n_components to 2 for dimensionality reduction
X_lda = lda.fit_transform(X_std, y)

# Step 4: Visualization (LDA results)
plt.figure(figsize=(10, 5))

# Plot LDA transformed data
plt.subplot(1, 2, 1)
for label, color in zip(np.unique(y), ['red', 'green', 'blue']):
    plt.scatter(X_lda[y == label, 0], X_lda[y == label, 1], color=color, label=iris.target_names[label], alpha=0.7)
plt.title("LDA: 2 Components")
plt.xlabel("LDA Component 1")
plt.ylabel("LDA Component 2")
plt.legend()

# Step 5: Compare LDA with PCA
# Apply PCA for comparison
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)

# Plot PCA transformed data
plt.subplot(1, 2, 2)
for label, color in zip(np.unique(y), ['red', 'green', 'blue']):
    plt.scatter(X_pca[y == label, 0], X_pca[y == label, 1], color=color, label=iris.target_names[label], alpha=0.7)
plt.title("PCA: 2 Components")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
