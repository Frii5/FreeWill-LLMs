import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# Load data
# -----------------------------
df_part1 = pd.read_csv("worths_by_model_part1.csv").set_index("model")
df_part2 = pd.read_csv("worths_by_model_part2.csv").set_index("model")

# -----------------------------
# Define item orders
# -----------------------------
item_order_part1 = [
    "FW1", "FW2", "FW3", "FW4", "FW5",
    "DE1", "DE2", "DE3", "DE4", "DE5",
    "DU1", "DU2", "DU3", "DU4", "DU5"
]

item_order_part2 = [
    "FC1", "FC2", "FC3", "FC4", "FC5", "FC6", "FC7",
    "MC1", "MC2", "MC3", "MC4", "MC5", "MC6", "MC7"
]

df_part1 = df_part1[item_order_part1]
df_part2 = df_part2[item_order_part2]

# -----------------------------
# Plot
# -----------------------------
fig, axes = plt.subplots(
    nrows=2,
    ncols=1,
    figsize=(14, 12),
    constrained_layout=True
)

# Part I
ax1 = sns.heatmap(
    df_part1,
    cmap="cividis",
    vmin=0,
    vmax=1,
    annot=True,
    fmt=".2f",
    ax=axes[0]
)
ax1.axvline(5, color="white", linewidth=1)
ax1.axvline(10, color="white", linewidth=1)
ax1.set_title("Part I: Plackett-Luce item worths by model")
ax1.set_xlabel("Item")
ax1.set_ylabel("Model")
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right")

# Part II
ax2 = sns.heatmap(
    df_part2,
    cmap="cividis",
    vmin=0,
    vmax=1,
    annot=True,
    fmt=".2f",
    ax=axes[1]
)
ax2.axvline(7, color="white", linewidth=1)
ax2.set_title("Part II: Plackett-Luce item worths by model")
ax2.set_xlabel("Item")
ax2.set_ylabel("Model")
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right")

plt.show()