# =======================================================================
# FUNCTIONS
# Define reusable helper functions in this file.
# =======================================================================

import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from src.constants import *
from src.constants import HOURS_PER_DAY


def load_batches():
    batches = {}

    for batch_id in SELECTED_BATCH_IDS:
        filename = f"batch_{batch_id:04d}.csv"
        filepath = os.path.join("datasets", filename)

        batches[batch_id] = pd.read_csv(filepath)

    return batches

def load_inital_conditions():
    filepath = os.path.join("datasets", "initial_conditions.csv")
    df = pd.read_csv(filepath)
    selected_conditions = []
    for batch_id in SELECTED_BATCH_IDS:
        df_batch = df[df["Batch ID"] == batch_id]
        selected_conditions.append(df_batch.iloc[0])

    initial_conditions = pd.DataFrame(selected_conditions)

    return initial_conditions

def prep_training_data(batches, initial_conditions):
    training_data = []

    for batch_id in batches:
        batch = batches[batch_id]
        df_initial_conditions = initial_conditions[initial_conditions["Batch ID"] == batch_id]

        initial_glucose = df_initial_conditions["Glucose [mM]"].iloc[0]
        initial_glutamine = df_initial_conditions["Glutamine [mM]"].iloc[0]
        initial_vcd = df_initial_conditions["VCD [cells/mL]"].iloc[0]

        for i in range(len(batch)):
            time = batch["Time [h]"].iloc[i]/HOURS_PER_DAY
            mab = batch["mAb [mg/L]"].iloc[i]

            training_data.append({"Batch ID": batch_id,
                                  "Initial Glucose [mM]": initial_glucose,
                                  "Initial Glutamine [mM]": initial_glutamine,
                                  "Initial VCD [cells/mL]": initial_vcd,
                                  "Harvest time [days]": time,
                                  "mAb [mg/L]": mab})

    return pd.DataFrame(training_data)

def plot_actual_vs_predicted(y_test, y_pred_poly, y_pred_knn):
    # Create figure with two subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), dpi=200, layout="constrained")

    # Creating the perfect prediction line
    lower = min(y_test.min(), y_pred_poly.min(), y_pred_knn.min())
    upper = max(y_test.max(), y_pred_poly.max(), y_pred_knn.max())

    # Plot poly actual vs predicted
    axes[0].scatter(y_test, y_pred_poly, color=COLORS[0], s=26, alpha=0.7,
                    edgecolors="black", linewidth=0.5, label="Model predictions")

    axes[0].plot([lower, upper], [lower, upper], color=COLORS[3],
                 linestyle=LINESTYLES[2], label="Perfect prediction")

    axes[0].set_title("Polynomial Regression (Degree 2)")

    # Plot knn actual vs predicted
    axes[1].scatter(y_test, y_pred_knn, color=COLORS[0], s=26, alpha=0.7,
                    edgecolors="black", linewidth=0.5, label="Model predictions")

    axes[1].plot([lower, upper], [lower, upper], color=COLORS[3],
                 linestyle=LINESTYLES[2], label="Perfect prediction")

    axes[1].set_title("KNN Regression (k=3, Uniform)")

    # Formatting subplots
    for ax in axes:
        # Setting common axis limits for easier comparison
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)

        ax.set_xlabel("Actual mAb Concentration [mg/L]")
        ax.set_ylabel("Predicted mAb Concentration [mg/L]")
        ax.legend(fontsize=fontsize_legend)

    # Export figure
    fig.savefig(os.path.join("figures", f"figure_01_actual_vs_predicted.png"))
    plt.close(fig)

def calculate_production_economics(glucose, glutamine, vcd, harvest_time, mab):
    # Calculate cell culture cost ($/L)
    culture_cost = 10 * glucose + 20 * glutamine + 0.0005 * vcd

    # Calculate cell culture value ($/L)
    mab_value = 1000 * mab

    # Calculate profit ($/L)
    profit = mab_value - culture_cost

    # Calculate space-time profit ($/L.day)
    space_time_profit = profit/(harvest_time + 1)

    # Calculate space-time yield (mg/L.day)
    space_time_yield = mab/(harvest_time + 1)

    return culture_cost, mab_value, profit, space_time_profit, space_time_yield

def generate_conditions():
    possible_conditions = []

    for glucose in range(5, 36, 5):
        for glutamine in range(1,8):
            for vcd in range(200000, 1400001, 200000):
                for time_in_hours in range(0, 673, 8):
                    harvest_time = time_in_hours/HOURS_PER_DAY

                    possible_conditions.append({"Initial Glucose [mM]": glucose,
                                       "Initial Glutamine [mM]": glutamine,
                                       "Initial VCD [cells/mL]": vcd,
                                       "Harvest time [days]": harvest_time,})

    return pd.DataFrame(possible_conditions)

def evaluate_conditions(possible_conditions, mab_predictions):
    results = []

    for i in range(len(possible_conditions)):
        glucose = possible_conditions.iloc[i]["Initial Glucose [mM]"]
        glutamine = possible_conditions.iloc[i]["Initial Glutamine [mM]"]
        vcd = possible_conditions.iloc[i]["Initial VCD [cells/mL]"]
        harvest_time = possible_conditions.iloc[i]["Harvest time [days]"]
        mab = mab_predictions[i]

        culture_cost, mab_value, profit, space_time_profit, space_time_yield \
            = calculate_production_economics(glucose, glutamine, vcd, harvest_time, mab)

        results.append({"Initial Glucose [mM]": glucose,
                        "Initial Glutamine [mM]": glutamine,
                        "Initial VCD [cells/mL]": vcd,
                        "Harvest time [days]": harvest_time,
                        "Predicted mAb [mg/L]": mab,
                        "Culture cost [$/L]": culture_cost,
                        "mAb value [$/L]": mab_value,
                        "Profit [$/L]": profit,
                        "Space-time profit [$/L.day]": space_time_profit,
                        "Space-time yield [mg/L.day]": space_time_yield})

    return pd.DataFrame(results)

def calculate_pred_interval(prediction, residuals, prediction_interval=0.95):
    alpha = 1 - prediction_interval

    # Calculate residual stats
    res_mean = np.mean(residuals)
    res_sd = np.std(residuals)

    # Calculate student's T critical value
    q = alpha/2
    dof = len(residuals) - 1
    pi_sd = stats.t.isf(q=q, df=dof)

    # Calculate prediction interval
    pi_lower = prediction - res_mean - pi_sd * res_sd
    pi_upper = prediction - res_mean + pi_sd * res_sd

    return pi_lower, pi_upper
