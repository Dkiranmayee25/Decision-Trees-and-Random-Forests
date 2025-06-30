import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv('heart.csv')
print("First 5 rows:\n", df.head())
print("\nDataset Info:\n")
print(df.info())

# Split features and target
X = df.drop('target', axis=1)
y = df['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree
dtree = DecisionTreeClassifier(random_state=42)
dtree.fit(X_train, y_train)

# Visualize the tree
plt.figure(figsize=(20,10))
plot_tree(dtree, feature_names=X.columns, class_names=["No Disease", "Disease"], filled=True)
plt.title("Decision Tree")
plt.show()

# Accuracy
print("\nDecision Tree Train Accuracy:", dtree.score(X_train, y_train))
print("Decision Tree Test Accuracy:", dtree.score(X_test, y_test))

# Decision Tree with limited depth
dtree_limited = DecisionTreeClassifier(max_depth=4, random_state=42)
dtree_limited.fit(X_train, y_train)
print("\nLimited Depth Decision Tree Test Accuracy:", dtree_limited.score(X_test, y_test))

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predictions and Evaluation
y_pred_rf = rf.predict(X_test)
print("\nRandom Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))

# Feature Importances
importances = pd.Series(rf.feature_importances_, index=X.columns)
importances.sort_values().plot(kind='barh', figsize=(10,6), title="Feature Importances (Random Forest),scaled")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()

# Cross-validation scores
dtree_cv = cross_val_score(dtree, X, y, cv=5)
rf_cv = cross_val_score(rf, X, y, cv=5)

print("\nCross-Validation Scores:")
print(f"Decision Tree Mean Accuracy: {dtree_cv.mean():.3f}")
print(f"Random Forest Mean Accuracy: {rf_cv.mean():.3f}")
