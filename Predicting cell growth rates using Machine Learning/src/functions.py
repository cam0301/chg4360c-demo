# =======================================================================
# FUNCTIONS
# Define reusable helper functions in this file.
# =======================================================================
import os

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from src.constants import *


def plot_poly_validation_curves(model_gscv):
    # Retrieve GridSearchCV results
    df = pd.DataFrame(model_gscv.cv_results_)

    # Retrieve polynomial degrees and RMSE values
    degrees = df["param_polynomialfeatures__degree"].to_numpy()
    rmse_train = -df["mean_train_score"].to_numpy()
    rmse_valid = -df["mean_test_score"].to_numpy()

    # Plot figure and format
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=200, layout="constrained")

    ax.plot(degrees, rmse_train, label="Train (Poly)", color=COLORS[0], linestyle=LINESTYLES[0])
    ax.plot(degrees, rmse_valid, label="Valid (Poly)", color=COLORS[0], linestyle=LINESTYLES[1])
    ax.set_xlabel("Polynomial Degree", fontsize=fontsize_xy_label)
    ax.set_ylabel("Specific Growth Rate RMSE [d^-1]", fontsize=fontsize_xy_label)
    ax.tick_params(axis='both', which='major', labelsize=fontsize_ticks)
    ax.legend(fontsize=fontsize_legend)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(MultipleLocator(1))

    # Export figure
    fig.savefig(os.path.join("figures", f"figure_01_validation_curves_poly.png"))
    plt.close(fig)

def plot_actual_vs_predicted(y_test, y_pred_poly):
    # Create figure
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=200)

    # Plot model predictions and format
    ax.scatter(y_test, y_pred_poly, color=COLORS[0], s=26, alpha=0.7,
               edgecolors="black", linewidth=0.5, label="Model predictions")

    # Creating the perfect prediction line
    lower = min(y_test.min(), y_pred_poly.min())
    upper = max(y_test.max(), y_pred_poly.max())

    # Plot perfect prediction and format
    ax.plot([lower, upper], [lower, upper], color=COLORS[3],
            linestyle=LINESTYLES[2], label="Perfect prediction")
    ax.set_xlabel("Actual Specific Growth Rate [d^-1]", fontsize=fontsize_xy_label)
    ax.set_ylabel("Predicted Specific Growth Rate [d^-1]", fontsize=fontsize_xy_label)
    ax.legend(fontsize=fontsize_legend)

    # Export figure
    fig.savefig(os.path.join("figures", f"figure_02_actual_vs_predicted.png"))
    plt.close(fig)

def plot_knn_validation_curves(model_gscv):
    # Retrieve GridSearchCV results
    df = pd.DataFrame(model_gscv.cv_results_)

    # Define weighting methods
    weights = ["uniform", "distance"]

    # Plot figure and format
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=200, layout="constrained")

    for i, weight in enumerate(weights):
        mask_weight = (df.loc[:, "param_kneighborsregressor__weights"] == weight)
        df_sub = df.loc[mask_weight, :]
        df_sub = df_sub.sort_values(by=['param_kneighborsregressor__n_neighbors'], ascending=True)
        n_neighbors = df_sub.loc[:, "param_kneighborsregressor__n_neighbors"].to_numpy()
        rmse_train = - df_sub.loc[:, "mean_train_score"].to_numpy()
        rmse_test = - df_sub.loc[:, "mean_test_score"].to_numpy()

        ax.plot(n_neighbors, rmse_train, label=f"Train (kNN: {weight})", color=COLORS[i], linestyle=LINESTYLES[0])
        ax.plot(n_neighbors, rmse_test, label=f"Valid (kNN: {weight})", color=COLORS[i], linestyle=LINESTYLES[1])

    ax.set_xlabel("Number of neighbors k", fontsize=fontsize_xy_label)
    ax.set_ylabel("Specific Growth Rate RMSE [d^-1]", fontsize=fontsize_xy_label)
    ax.tick_params(axis='both', which='major', labelsize=fontsize_ticks)
    ax.legend(fontsize=fontsize_legend)
    ax.set_ylim(bottom=0)

    # Export figure
    fig.savefig(os.path.join("figures", "figure_03_validation_curve_knn.png"))
    plt.close(fig)

def plot_model_comparison(y_test, y_pred_poly, y_pred_knn):
    # Create figure with two subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), dpi=200, layout="constrained")

    # Creating the perfect prediction line
    lower = min(y_test.min(), y_pred_poly.min(), y_pred_knn.min())
    upper = max(y_test.max(), y_pred_poly.max(), y_pred_knn.max())

    # Plot poly actual vs predicted
    axes[0].scatter(y_test, y_pred_poly,color=COLORS[0], s=26, alpha=0.7,
                    edgecolors="black", linewidth=0.5, label="Model predictions")

    axes[0].plot([lower, upper], [lower, upper], color=COLORS[3],
                 linestyle=LINESTYLES[2], label="Perfect prediction")

    axes[0].set_title("Polynomial Regression (Degree 2)")

    # Plot knn actual vs predicted
    axes[1].scatter(y_test, y_pred_knn, color=COLORS[0], s=26, alpha=0.7,
                    edgecolors="black", linewidth=0.5, label="Model predictions")

    axes[1].plot([lower, upper], [lower, upper], color=COLORS[3],
                 linestyle=LINESTYLES[2], label="Perfect prediction")

    axes[1].set_title("KNN Regression (k=9, Distance)")

    # Formatting subplots
    for ax in axes:
        # Setting common axis limits for easier comparison
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)

        ax.set_xlabel("Actual Specific Growth Rate [d^-1]")
        ax.set_ylabel("Predicted Specific Growth Rate [d^-1]")
        ax.legend(fontsize=fontsize_legend)

    # Export figure
    fig.savefig(os.path.join("figures", f"figure_04_model_comparison.png"))
    plt.close(fig)

def plot_variables_vs_growth(model_gscv, x_train, feature_names):
    # Create figure with four subplots
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 7), dpi=200, layout="constrained")

    # Calculate the mean of each input variable
    mean_values = np.mean(x_train, axis=0)

    # Loop through the rows and columns of the figure
    for row in range(2):
        for col in range(2):
            # Identify the feature being investigated
            i = row * 2 + col
            feature = feature_names[i]

            # Generate 100 values across the training range
            variable_values = np.linspace(
                x_train[:, i].min(),
                x_train[:, i].max(),
                100)

            # Create operating conditions while varying one variable
            x_axis = []
            for value in variable_values:
                variables = mean_values.copy()
                variables[i] = value
                x_axis.append(variables)
            x_axis = np.array(x_axis)

            # Predict cell growth rate
            y_axis = model_gscv.predict_targets(x_axis)

            # Plot and format predicted growth rate
            axes[row,col].plot(variable_values, y_axis)

            axes[row,col].set_xlabel(feature)
            axes[row,col].set_ylabel("Predicted Growth Rate [d^-1]")
            axes[row,col].set_title(f"Effect of {feature}")

    # Export figure
    fig.savefig(os.path.join("figures", "figure_05_operating_conditions.png"))
    plt.close(fig)
