# =====================================================================
# MAIN
# =====================================================================
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import src.functions as sf
from src.classes import MabProductionPolyModel, MabProductionKNNModel
from src.constants import *

# ============================================================================
# DATA LOADING AND PRE-PROCESSING
# ============================================================================

# Load batches
batches = sf.load_batches()

# print(f"Number of batches: {len(batches)}")

for batch_id in batches:
    batch = batches[batch_id]
    # print(f"Batch {batch_id:04d}: {len(batch)} data")

# Load intial conditions
initial_conditions = sf.load_inital_conditions()

# print(f"\nInitial conditions:\n {initial_conditions}")

# Prepare training data
training_data = sf.prep_training_data(batches, initial_conditions)

# print(f"\nTraining data:\n {training_data}")
# print(f"\nNumber of observations:\n {len(training_data)}")

# Define input and output variables
feature_names = ["Initial Glucose [mM]",
                 "Initial Glutamine [mM]",
                 "Initial VCD [cells/mL]",
                 "Harvest time [days]"]

x = training_data[feature_names].to_numpy()
y = training_data["mAb [mg/L]"].to_numpy()

# Split batch IDs into training and testing sets
training_ids, testing_ids = train_test_split(SELECTED_BATCH_IDS, test_size=0.2, random_state=0)

# Create masks to identify training and testing observations
train_mask = training_data["Batch ID"] == training_ids[0]
test_mask = training_data["Batch ID"] == testing_ids[0]

for batch_id in training_ids[1:]:
    train_mask = train_mask | (training_data["Batch ID"] == batch_id)

for batch_id in testing_ids[1:]:
    test_mask = test_mask | (training_data["Batch ID"] == batch_id)

# Separate training and testing data
x_train = x[train_mask]
y_train = y[train_mask]
x_test = x[test_mask]
y_test = y[test_mask]

# Batch IDs for each training observation
groups = training_data.loc[train_mask, "Batch ID"].to_numpy()

# ============================================================================
# POLYNOMIAL REGRESSION
# ============================================================================
print(f" ============================================================================"
      f" POLYNOMIAL REGRESSION"
      f" ============================================================")


# Create and train polynomial model
model = MabProductionPolyModel()
model.optimal_hyperparam(x_train, y_train, groups)

# Evaluate model
print(f"\nFinal Test negative RMSE Poly: {model.evaluate_model(x_test, y_test):.6f} mg/L\n")

# ============================================================================
# KNN REGRESSION
# ============================================================================
print(f" ============================================================================"
      f" KNN REGRESSION"
      f" ===================================================================")

# Create and train KNN model
knn_model = MabProductionKNNModel()
knn_model.optimal_hyperparam(x_train, y_train, groups)

# Evaluate model
print(f"\nFinal Test negative RMSE KNN: {knn_model.evaluate_model(x_test, y_test):.6f} mg/L\n")

# ============================================================================
# MODEL COMPARISON
# ============================================================================

# Predict mAb concentrations on testing data
y_pred_poly = model.predict_targets(x_test)
y_pred_knn = knn_model.predict_targets(x_test)

# Compare models
sf.plot_actual_vs_predicted(y_test, y_pred_poly, y_pred_knn)

# ============================================================================
# ECONOMICS
# ============================================================================
print(f" ========================================================================="
      f" OPTIMIZATION"
      f" ========================================================================")

# Generate candidate operating conditions
possible_conditions = sf.generate_conditions()
# print(f"\nPossible operating conditions: {possible_conditions}")
print(f"Number of possible conditions: {len(possible_conditions)}")

# Retrieve candidate input variables
x_conditions = possible_conditions.to_numpy()

# Predict and compare mAb concentrations for both models
mab_pred_poly = model.predict_targets(x_conditions)
mab_pred_knn = knn_model.predict_targets(x_conditions)

# Check for negative predictions
print("\nMinimum predicted mAb concentration:")
print(f"Polynomial: {np.min(mab_pred_poly):.6f} mg/L")
print(f"kNN: {np.min(mab_pred_knn):.6f} mg/L")

print("\nNumber of negative predictions:")
print("Polynomial:", np.sum(mab_pred_poly < 0))
print("kNN:", np.sum(mab_pred_knn < 0))


# Evaluate the models
results_poly = sf.evaluate_conditions(possible_conditions, mab_pred_poly)
results_knn = sf.evaluate_conditions(possible_conditions, mab_pred_knn)

# Determine optimal conditions for maximized space-time profit
index_optimal_poly = np.argmax(results_poly["Space-time profit [$/L.day]"].to_numpy())
index_optimal_knn = np.argmax(results_knn["Space-time profit [$/L.day]"].to_numpy())

print(f"\nOptimal conditions according to Polynomial regression:")
print(results_poly.iloc[index_optimal_poly].to_string())
print(f"\nOptimal conditions according to KNN regression:")
print(results_knn.iloc[index_optimal_knn].to_string())

# Export results
results_poly.to_csv(
    os.path.join("tables", "table_03_poly_optimization.csv"), index=False)

results_knn.to_csv(os.path.join("tables", "table_04_knn_optimization.csv"), index=False)

# Calculate residual error on testing data
res_poly = y_pred_poly - y_test
res_knn = y_pred_knn - y_test

print(f"\nPolynomial residual mean:", np.mean(res_poly))
print(f"Polynomial residual SD:", np.std(res_poly))
print(f"\nKNN residual mean:", np.mean(res_knn))
print(f"nKNN residual SD:", np.std(res_knn))

# Retrieve optimal predicted mAb concentrations
optimal_mab_poly = results_poly.iloc[index_optimal_poly]["Predicted mAb [mg/L]"]
optimal_mab_knn = results_knn.iloc[index_optimal_poly]["Predicted mAb [mg/L]"]

# Calculate mAb prediction intervals
pi_poly_lower, pi_poly_upper = sf.calculate_pred_interval(optimal_mab_poly, res_poly)
pi_knn_lower, pi_knn_upper = sf.calculate_pred_interval(optimal_mab_knn, res_knn)

# Retrieve optimal conditions
optimal_poly = results_poly.iloc[index_optimal_poly]
optimal_knn = results_knn.iloc[index_optimal_knn]

# Calculate economic intervals
_, poly_mab_value_l, _, poly_stp_l, poly_sty_l = sf.calculate_production_economics(
    optimal_poly["Initial Glucose [mM]"],
    optimal_poly["Initial Glutamine [mM]"],
    optimal_poly["Initial VCD [cells/mL]"],
    optimal_poly["Harvest time [days]"],
    pi_poly_lower
)

_, poly_mab_value_u, _, poly_stp_u, poly_sty_u = sf.calculate_production_economics(
    optimal_poly["Initial Glucose [mM]"],
    optimal_poly["Initial Glutamine [mM]"],
    optimal_poly["Initial VCD [cells/mL]"],
    optimal_poly["Harvest time [days]"],
    pi_poly_upper
)

_, knn_mab_value_l, _, knn_stp_l, knn_sty_l = sf.calculate_production_economics(
    optimal_knn["Initial Glucose [mM]"],
    optimal_knn["Initial Glutamine [mM]"],
    optimal_knn["Initial VCD [cells/mL]"],
    optimal_knn["Harvest time [days]"],
    pi_knn_lower
)

_, knn_mab_value_u, _, knn_stp_u, knn_sty_u = sf.calculate_production_economics(
    optimal_knn["Initial Glucose [mM]"],
    optimal_knn["Initial Glutamine [mM]"],
    optimal_knn["Initial VCD [cells/mL]"],
    optimal_knn["Harvest time [days]"],
    pi_knn_upper
)

print(f"\nPolynomial regression 95% prediction intervals:")
print(f"mAb:{pi_poly_lower:.6f} to {pi_poly_upper:.6f}")
print(f"Space-time profit: {poly_stp_l:.6f} {poly_stp_u:.6f}")
print(f"Space-time yield: {poly_sty_l:.6f} {poly_sty_u:.6f}")
print(f"\nKNN regression 95% prediction intervals:")
print(f"mAb: {pi_knn_lower:.6f} {pi_knn_upper:.6f}")
print(f"Space-time profit: {knn_stp_l:.6f} {knn_stp_u:.6f}")
print(f"Space-time yield: {knn_sty_l:.6f} {knn_sty_u:.6f}")


# Create optimization summary
summary = []

summary.append({
    "Model": "Polynomial",
    "Glucose [mM]": optimal_poly["Initial Glucose [mM]"],
    "Glutamine [mM]": optimal_poly["Initial Glutamine [mM]"],
    "VCD [cells/mL]": optimal_poly["Initial VCD [cells/mL]"],
    "Harvest time [days]": optimal_poly["Harvest time [days]"],
    "Predicted mAb [mg/L]": optimal_poly["Predicted mAb [mg/L]"],
    "Space-time profit [$/L.day]": optimal_poly["Space-time profit [$/L.day]"],
    "Profit PI lower": poly_stp_l,
    "Profit PI upper": poly_stp_u,
    "Yield PI lower": poly_sty_l,
    "Yield PI upper": poly_sty_u
})

summary.append({
    "Model": "kNN",
    "Glucose [mM]": optimal_knn["Initial Glucose [mM]"],
    "Glutamine [mM]": optimal_knn["Initial Glutamine [mM]"],
    "VCD [cells/mL]": optimal_knn["Initial VCD [cells/mL]"],
    "Harvest time [days]": optimal_knn["Harvest time [days]"],
    "Predicted mAb [mg/L]": optimal_knn["Predicted mAb [mg/L]"],
    "Space-time profit [$/L.day]": optimal_knn["Space-time profit [$/L.day]"],
    "Profit PI lower": knn_stp_l,
    "Profit PI upper": knn_stp_u,
    "Yield PI lower": knn_sty_l,
    "Yield PI upper": knn_sty_u
})

df_summary = pd.DataFrame(summary)

df_summary.to_csv(
    os.path.join("tables", "table_05_optimization_summary.csv"),
    index=False
)

print("\nOptimization summary:")
print(df_summary.to_string(index=False))

