from utils import db_connect
engine = db_connect()

# your code here

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# -------------------------------------
# 1. Load and inspect dataset
# -------------------------------------
url = 'https://raw.githubusercontent.com/4GeeksAcademy/regularized-linear-regression-project-tutorial/main/demographic_health_data.csv'
df = pd.read_csv(url)

print("Columns in dataset:")
print(df.columns.tolist())

# -------------------------------------
# 2. Define target variable
# -------------------------------------
target = 'Obesity_prevalence'

if target not in df.columns:
    raise ValueError(f"Target column '{target}' does not exist in the dataset. Please choose a valid target.")

# -------------------------------------
# 3. Data cleaning and preprocessing
# -------------------------------------

# Drop irrelevant columns
cols_to_drop = ['fips', 'COUNTY_NAME', 'STATE_NAME', 'STATE_FIPS', 'CNTY_FIPS']
cols_to_drop = [col for col in cols_to_drop if col in df.columns]
df.drop(columns=cols_to_drop, inplace=True)

# Drop columns with more than 30% missing values
threshold = len(df) * 0.3
cols_before = df.shape[1]
df = df.loc[:, df.isnull().sum() <= threshold]
cols_after = df.shape[1]
print(f"\nDropped {cols_before - cols_after} columns with >30% missing values")

if target not in df.columns:
    raise ValueError(f"Target column '{target}' was dropped due to missing values or not found! Adjust threshold or target.")

# Drop columns that leak target info (highly correlated/confounding)
leakage_cols = [
    'Obesity_Upper 95% CI',
    'Obesity_Lower 95% CI',
    'anycondition_Upper 95% CI',
    'anycondition_Lower 95% CI',
    'anycondition_prevalence'
]
leakage_cols = [col for col in leakage_cols if col in df.columns]
df.drop(columns=leakage_cols, inplace=True)

# Fill remaining missing numeric values with column mean
df.fillna(df.mean(numeric_only=True), inplace=True)

# -------------------------------------
# 4. Prepare features and target
# -------------------------------------
X = df.drop(columns=[target])
y = df[target]

# -------------------------------------
# 5. Split data into training and testing sets
# -------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------------
# 6. Train baseline Linear Regression model
# -------------------------------------
lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)
r2_lr = r2_score(y_test, y_pred_lr)
print(f"\nLinear Regression R2 score: {r2_lr:.4f}")

# -------------------------------------
# 7. Train and evaluate Lasso Regression over a range of alphas
# -------------------------------------
alphas = np.linspace(0, 20, 40)
r2_scores = []

for alpha in alphas:
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train, y_train)
    y_pred_lasso = lasso.predict(X_test)
    r2_scores.append(r2_score(y_test, y_pred_lasso))

# -------------------------------------
# 8. Plot R2 scores vs alpha values
# -------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(alphas, r2_scores, marker='o')
plt.title('Lasso Regression: R2 Score vs Alpha')
plt.xlabel('Alpha')
plt.ylabel('R2 Score')
plt.grid(True)
plt.show()

# -------------------------------------
# 9. Select best alpha and train final Lasso model
# -------------------------------------
best_alpha = alphas[np.argmax(r2_scores)]
print(f"\nBest alpha for Lasso: {best_alpha:.2f} with R2 = {max(r2_scores):.4f}")

lasso_final = Lasso(alpha=best_alpha, max_iter=10000)
lasso_final.fit(X_train, y_train)

y_pred_final = lasso_final.predict(X_test)
r2_final = r2_score(y_test, y_pred_final)
print(f"\nFinal Lasso model R2 score: {r2_final:.4f}")

# -------------------------------------
# 10. Cross-validation on Linear Regression for comparison
# -------------------------------------
cv_scores = cross_val_score(lr, X, y, cv=5, scoring='r2')
print(f"\nCross-validated R2 scores: {cv_scores}")
print(f"Mean CV R2 score: {cv_scores.mean():.4f}")
