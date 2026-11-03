# =======================================================================
# CONSTANTS
# Define constant values that do not change throughout your program.
# =======================================================================


# Dataset settings
SELECTED_BATCH_IDS = [6, 16, 49, 63, 86, 100, 138, 192, 289, 315,
                    326, 398, 442, 504, 510, 538, 548, 575, 586, 599,
                    643, 740, 755, 828, 839, 897, 930, 945, 980, 988]
TOTAL_BATCH_COUNT = 30

# Column names
BATCH_ID = "Batch ID"
TIME = "Time [h]"
GLUCOSE = "Glucose [mM]"
GLUTAMINE = "Gutamine [mM]"
LACTATE = "Lactate [mM]"
AMMONIA = "Ammonia [mM]"
VCD = "VCD [cells/mL]"
TCD = "TCD [cells/mL]"
MAB = "mAb [mg/L]"

# Economic parameters
GLUCOSE_COST = 10.0
GLUTAMINE_COST = 20.0
VCD_COST = 0.0005
MAB_VALUE = 1000

# Process parameters
HOURS_PER_DAY = 24.0
DOWNTIME_DAYS = 1.0

# Plot parameters
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
LINESTYLES = ["-", ":", "--", "-."]
FIG_WIDTH = 14.39 / 2.54
FIG_HEIGHT = 12.09 / 2.54
fontsize_xy_label = 14
fontsize_ticks = 12
fontsize_legend = 12
