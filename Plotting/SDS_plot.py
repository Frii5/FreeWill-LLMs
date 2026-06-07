from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import colors as mcolors

ROOT = Path(__file__).resolve().parent.parent

# ---------- font ----------
plt.rcParams["font.family"] = "Times New Roman"      
plt.rcParams["font.size"] = 20

# ---------- helper ----------
def darken(color, factor=0.72):
    r, g, b = mcolors.to_rgb(color)
    return (r * factor, g * factor, b * factor)

def lighten(color, factor=1.18):
    r, g, b = mcolors.to_rgb(color)
    return (
        min(r * factor, 1.0),
        min(g * factor, 1.0),
        min(b * factor, 1.0),
    )

# ---------- data ----------
part1 = pd.read_csv(ROOT / "results_SDS/sds_scores_part1.csv")
part2 = pd.read_csv(ROOT / "results_SDS/sds_scores_part2.csv")

part1["dimension"] = ["FW"] * 5 + ["DE"] * 5 + ["DU"] * 5
part2["dimension"] = ["FC"] * 7 + ["MC"] * 7

df = pd.concat([part1, part2], ignore_index=True)
df["item_id"] = range(1, len(df) + 1)
df["short_label"] = (
    df.groupby("dimension")
      .cumcount()
      .add(1)
      .astype(str)
)
df["short_label"] = df["dimension"] + df["short_label"]

colors = {
    "FW": "#4C72B0",
    "DE": "#55A868",
    "DU": "#C44E52",
    "FC": "#8172B3",
    "MC": "#DD8452",
}

labels = {
    "FW": "FW",
    "DE": "DE",
    "DU": "DU",
    "FC": "FC",
    "MC": "MC",
}

# ---------- plot ----------
fig, ax = plt.subplots(figsize=(14, 6))

ax.bar(
    df["item_id"],
    df["med"],
    color=[colors[d] for d in df["dimension"]],
    edgecolor=[darken(colors[d], 0.7) for d in df["dimension"]],
    linewidth=1.5,
    width=0.8,
    zorder=2
)

# dashed mean lines per dimension
for dim in ["FW", "DE", "DU", "FC", "MC"]:
    sub = df[df["dimension"] == dim]
    y = sub["med"].mean()
    x0 = sub["item_id"].min() - 0.4
    x1 = sub["item_id"].max() + 0.4

    ax.hlines(
        y, x0, x1,
        color=darken(colors[dim], 0.68),
        linewidth=1.5,
        linestyles=(0, (4, 3)),   # dashed
        alpha=1.0,
        zorder=5                  # float on top
    )

#ax.set_xlabel("Item")
ax.set_ylabel("Social Desirability Score")
ax.set_xlim(0.3, len(df) + 0.7)
ax.set_ylim(0, max(10, df["med"].max() + 0.5))

ax.set_xticks(df["item_id"])
ax.set_xticklabels(df["short_label"], rotation=55, ha="right", rotation_mode="anchor")

handles = [Patch(facecolor=colors[k], label=labels[k]) for k in ["FW", "DE", "DU", "FC", "MC"]]
ax.legend(handles=handles, ncol=5, loc="upper center", frameon=False)

plt.tight_layout()
plt.savefig("Plotting/sds_barplot.svg", bbox_inches="tight")
plt.show()