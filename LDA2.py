# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
import seaborn as sns

# Step 1: Load the Dataset
wine = load_wine()
X = wine.data  # features
y = wine.target  # target labels

# Step 2: Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Train an LDA Model
lda = LDA()
lda.fit(X_train, y_train)

# Step 4: Evaluate the Model
# Predict the labels on the test set
y_pred_lda = lda.predict(X_test)

# Calculate accuracy, precision, recall, and confusion matrix
accuracy_lda = accuracy_score(y_test, y_pred_lda)
precision_lda = precision_score(y_test, y_pred_lda, average='weighted')
recall_lda = recall_score(y_test, y_pred_lda, average='weighted')
conf_matrix_lda = confusion_matrix(y_test, y_pred_lda)

print("LDA Classifier Performance:")
print(f"Accuracy: {accuracy_lda:.2f}")
print(f"Precision: {precision_lda:.2f}")
print(f"Recall: {recall_lda:.2f}")
print("\nConfusion Matrix:\n", conf_matrix_lda)
print("\nClassification Report:\n", classification_report(y_test, y_pred_lda))

# Step 5: Compare with Logistic Regression
log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)

# Calculate metrics for Logistic Regression
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
precision_log_reg = precision_score(y_test, y_pred_log_reg, average='weighted')
recall_log_reg = recall_score(y_test, y_pred_log_reg, average='weighted')
conf_matrix_log_reg = confusion_matrix(y_test, y_pred_log_reg)

print("\nLogistic Regression Classifier Performance:")
print(f"Accuracy: {accuracy_log_reg:.2f}")
print(f"Precision: {precision_log_reg:.2f}")
print(f"Recall: {recall_log_reg:.2f}")
print("\nConfusion Matrix:\n", conf_matrix_log_reg)
print("\nClassification Report:\n", classification_report(y_test, y_pred_log_reg))

# Step 6: Visualize Decision Boundaries (Optional)
# Use LDA to reduce the dataset to 2 dimensions for visualization purposes
lda_2d = LDA(n_components=2)
X_lda_2d = lda_2d.fit_transform(X, y)

# Scatter plot for LDA
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for label, color in zip(np.unique(y), ['red', 'green', 'blue']):
    plt.scatter(X_lda_2d[y == label, 0], X_lda_2d[y == label, 1], color=color, label=wine.target_names[label], alpha=0.7)
plt.title("LDA Decision Boundary Visualization")
plt.xlabel("LDA Component 1")
plt.ylabel("LDA Component 2")
plt.legend()

# Scatter plot for Logistic Regression
pca = PCA(n_components=2)
X_pca_2d = pca.fit_transform(X)

plt.subplot(1, 2, 2)
for label, color in zip(np.unique(y), ['red', 'green', 'blue']):
    plt.scatter(X_pca_2d[y == label, 0], X_pca_2d[y == label, 1], color=color, label=wine.target_names[label], alpha=0.7)
plt.title("Logistic Regression Decision Boundary Visualization")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()

plt.tight_layout()
plt.show()
