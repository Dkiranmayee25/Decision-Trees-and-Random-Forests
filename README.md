# Decision-Trees-and-Random-Forests
Heart Disease Prediction using Decision Tree and Random Forest

Objective
Build and compare Decision Tree and Random Forest classifiers to predict the presence of heart disease based on health-related attributes.

Dataset
- **Source**: heart.csv
- Features include age, sex, chest pain type, resting blood pressure, cholesterol, etc.
- Target: `0` (No Disease), `1` (Disease)

Tools Used
- Python
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn

Tasks Performed
1. Load and clean the dataset
2. Train a Decision Tree Classifier and visualize the tree
3. Analyze overfitting by controlling tree depth
4. Train a Random Forest Classifier and evaluate
5. Interpret feature importances
6. Evaluate performance using accuracy, classification report, and cross-validation

Results
- Decision Tree Test Accuracy: 0.9853658536585366
- Limited Depth Decision Tree Accuracy: 0.8
- Random Forest Accuracy:  0.9853658536585366
- Cross-Validation Accuracy: Decision Tree Mean Accuracy: 1.000
                             Random Forest Mean Accuracy: 0.997

Feature Importances
Random Forest revealed that features like `cp`, `thalach`, and `ca` have the most impact on predictions.
