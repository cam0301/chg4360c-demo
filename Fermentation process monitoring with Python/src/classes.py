import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


class BioprocessMonitor:
    def __init__(self, filepath, ph_lims, temperature_lims):
        self.df = pd.read_csv(filepath)
        self.ph_lims = ph_lims
        self.temperature_lims = temperature_lims

    def extract_batch(self, batch_id):
        return self.df[self.df["batch_id"] == batch_id]

    def optimal_ph_mask(self, df_batch):
        pH_min = self.ph_lims[0]
        pH_max = self.ph_lims[1]
        mask_pH = (df_batch["pH"] >= pH_min) & (df_batch["pH"] <= pH_max)
        return mask_pH

    def optimal_temperature_mask(self, df_batch):
        temp_min = self.temperature_lims[0]
        temp_max = self.temperature_lims[1]
        mask_temp = (df_batch["temperature_C"] >= temp_min) & (df_batch["temperature_C"] <= temp_max)
        return mask_temp

    def get_n_batches(self):
        all_batches = self.df["batch_id"].to_numpy()
        unique_batches = np.unique(all_batches)
        return len(unique_batches)

    def export_dashboard(self, batch_id, filepath):
        df_batch = self.extract_batch(batch_id)
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6.5, 4), dpi=200, layout="constrained")
        axes[0, 0].scatter(
            df_batch["time_h"],
            df_batch["C_glucose_g_L^-1"],
            color="tab:blue", marker="o", s=26,
            alpha=0.7, edgecolors="black", linewidth=0.5,
            label="Glucose"
        )
        axes[0, 0].scatter(
            df_batch["time_h"],
            df_batch["C_biomass_g_L^-1"],
        color = "tab:orange", marker = "^", s = 26,
        alpha = 0.7, edgecolors = "black", linewidth=0.5,
        label="Biomass"
        )
        axes[0, 0].scatter(
            df_batch["time_h"],
            df_batch["C_product_g_L^-1"],
            color="tab:green", marker="s", s=26,
            alpha=0.7, edgecolors="black", linewidth=0.5,
            label="Product"
        )

        axes[0, 0].set_xlabel("Time (h)", fontsize=10)
        axes[0, 0].set_ylabel("Concentration (g/L)", fontsize=10)
        axes[0, 0].tick_params(axis='both', which='major', labelsize=10)
        axes[0, 0].legend(fontsize=10)

        time = df_batch["time_h"].to_numpy()
        temperature = df_batch["temperature_C"].to_numpy()
        temperature_mask = self.optimal_temperature_mask(df_batch)

        axes[0, 1].scatter(
            time[temperature_mask],
            temperature[temperature_mask],
            color="tab:green", marker="o", s=26,
            alpha=0.7, edgecolors="black", linewidth=0.5,
            label="Optimal"
        )
        axes[0, 1].scatter(
            time[~temperature_mask],
            temperature[~temperature_mask],
            color="tab:red", marker="X", s=26,
            alpha=0.7, edgecolors="black", linewidth=0.5,
            label="Sub-Optimal"
        )

        axes[0, 1].set_xlabel("Time (h)", fontsize=10)
        axes[0, 1].set_ylabel("Temperature (ºC)", fontsize=10)
        axes[0, 1].tick_params(axis='both', which='major', labelsize=10)
        axes[0, 1].legend(fontsize=10)

        pH = df_batch["pH"].to_numpy()
        pH_mask = self.optimal_ph_mask(df_batch)

        axes[1, 0].scatter(
            time[pH_mask],
            pH[pH_mask],
            color="tab:green", marker="o", s=26,
            alpha=0.7, edgecolors="black", linewidth=0.5,
            label="Optimal"
        )
        axes[1, 0].scatter(
            time[~pH_mask],
            pH[~pH_mask],
            color="tab:red", marker="X", s=26,
            alpha=0.7, edgecolors="black", linewidth=0.5,
            label="Sub-Optimal"
        )

        axes[1, 0].set_xlabel("Time (h)", fontsize=10)
        axes[1, 0].set_ylabel("pH", fontsize=10)
        axes[1, 0].tick_params(axis='both', which='major', labelsize=10)
        axes[1, 0].legend(fontsize=10)

        axes[1, 1].scatter(
            df_batch["time_h"],
            df_batch["DO_percent"],
            color="tab:blue", marker="o", s=26,
            alpha=0.7, edgecolors="black", linewidth=0.5,
        )

        axes[1, 1].set_xlabel("Time (h)", fontsize=10)
        axes[1, 1].set_ylabel("Dissolved Oxygen (DO) [%]", fontsize=10)
        axes[1, 1].tick_params(axis='both', which='major', labelsize=10)

        for ax in np.ravel(axes):
            ax.xaxis.set_major_locator(MultipleLocator(6))

        fig.savefig(filepath)
        plt.close(fig)

    def export_summary(self, filepath):
        unique_batches = np.unique(self.df["batch_id"].to_numpy())
        summary_table = []
        for batch_id in unique_batches:
            df_batch = self.extract_batch(batch_id)
            pH_mask = self.optimal_ph_mask(df_batch)
            pH_optimal_percent = round(np.sum(pH_mask)/len(pH_mask)*100, 2)
            temperature_mask = self.optimal_temperature_mask(df_batch)
            temperature_optimal_percent = round(np.sum(temperature_mask)/len(temperature_mask)*100, 2)
            df_batch.sort_values("time_h")
            product_final_concentration = df_batch["C_product_g_L^-1"].iloc[-1]
            summary_table.append({"batch_id":batch_id,
                                  "pH_optimal_percent":pH_optimal_percent,
                                  "temperature_optimal_percent":temperature_optimal_percent,
                                  "C_product_g_L^-1_final":product_final_concentration
            })
            summary_df = pd.DataFrame(summary_table)
            summary_df.to_csv(filepath, index=False)

