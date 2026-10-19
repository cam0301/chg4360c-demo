# =====================================================================
# MAIN PROGRAM
# Your code starts here. This is the file that will be run.
# =====================================================================

import os

import pandas as pd
from src.classes import CellGrowthModel, KNNComparisonModel
import src.functions as sf
from sklearn.model_selection import GridSearchCV, train_test_split

#Import data set
df = pd.read_csv(os.path.join("datasets", "dataset_cell_growth_4factor.csv"))

#Identify the features and the target
features = df.columns[1:]
targets = df.columns[0]
x = df.loc[:, features].to_numpy()
y = df.loc[:, targets].to_numpy()

#Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# ============================================================================
# POLYNOMIAL REGRESSION
# ============================================================================
print(f" ============================================================================"
      f" POLYNOMIAL REGRESSION"
      f" ============================================================")

# Create CellGrowthModel object
cell_growth_model = CellGrowthModel()

# Optimize the polynomial degree
cell_growth_model.optimal_hyperparam(x_train, y_train)

# Print the final test score on data that was never seen by GridSearchCV
print(f"\nFinal Test negative RMSE Poly: {cell_growth_model.evaluate_model(x_test, y_test):.6f} d^-1\n")

# Plot validation curves
sf.plot_poly_validation_curves(cell_growth_model.model_gscv)

# Predict specific growth rates
y_pred_poly = cell_growth_model.predict_targets(x_test)

# Plot actual versus predicted growth rates
sf.plot_actual_vs_predicted(y_test, y_pred_poly)

# ============================================================================
# KNN REGRESSION
# ============================================================================
print(f" ============================================================================"
      f" KNN REGRESSION"
      f" ===================================================================")

# Create KNNComparisonModel object
knn_model = KNNComparisonModel()

# Optimize the hyperparameters degree
knn_model.optimal_hyperparam(x_train, y_train)

# Print the final test score on data that was never seen by GridSearchCV
print(f"\nFinal Test negative RMSE KNN: {knn_model.evaluate_model(x_test, y_test):.6f} d^-1")

# Plot KNN validation curves
sf.plot_knn_validation_curves(knn_model.model_gscv)

# Predict specific growth rates
y_pred_knn = knn_model.predict_targets(x_test)

# Compare poly and knn by plotting actual vs predicted figure
sf.plot_model_comparison(y_test, y_pred_poly, y_pred_knn)

# ============================================================================
# STUDY ON EFFECT OF EACH VARIABLE
# ============================================================================
# Plot each variable vs cell growth
sf.plot_variables_vs_growth(cell_growth_model, x_train, features)