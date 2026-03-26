import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("worths_by_model.csv").set_index("model")

item_order = [
    "FW1", "FW2", "FW3", "FW4", "FW5",
    "DE1", "DE2", "DE3", "DE4", "DE5",
    "DU1", "DU2", "DU3", "DU4", "DU5"
]
df = df[item_order]

plt.figure(figsize=(12, 6))
ax = sns.heatmap(
    df,
    cmap="viridis",
    vmin=0,
    vmax=1
)

ax.axvline(5, color="white", linewidth=1)
ax.axvline(10, color="white", linewidth=1)

plt.title("Plackett-Luce item worths by model")
plt.xlabel("Item")
plt.ylabel("Model")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()