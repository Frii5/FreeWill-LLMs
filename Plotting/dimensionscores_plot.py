from pathlib import Path
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from part1_scoring import analyze_model_pkl as analyze_part1
from part2_scoring import analyze_model_pkl as analyze_part2


# ----------------------------
# config
# ----------------------------

plt.rcParams["font.family"] = "Palatino Linotype"      
plt.rcParams["font.size"] = 12
plt.rcParams["font.weight"] = "bold"

OUTFILE = "Plotting/bootstrapped_dimensions.svg"
N_BOOT = 5000

MODEL_NAME_MAP = {
    "claude-haiku-4-5": "Claude Haiku 4.5",
    "claude-opus-4-6": "Claude Opus 4.6",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "deepseek-chat": "DeepSeek Chat",
    "gemini-3-flash-preview": "Gemini 3 Flash",
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "google_gemma-3-4b": "Gemma 3 4B",
    "gpt-4o-2024-08-06": "GPT-4o",
    "gpt-5.4-mini": "GPT-5.4 Mini",
    "gpt-5.4-nano": "GPT-5.4 Nano",
    "gpt-5.4": "GPT-5.4",
    "grok-4-1-fast-reasoning": "Grok 4.1 Fast",
    "grok-4.20-0309-reasoning": "Grok 4.20",
    "meta-llama-3-8b-instruct": "Llama 3 8B",
    "mistral-large-2512": "Mistral Large",
    "mistral-medium-2508": "Mistral Medium",
    "nvidia_nemotron-3-nano": "Nemotron 3 Nano",
    "openai_gpt-oss-20b": "GPT-OSS 20B",
    "phi-4-reasoning-vision-15b": "Phi-4 RV 15B",
    "qwen_qwen3-vl-8b": "Qwen3-VL 8B",
}

COLORS_1 = {
    "FW": "#4C72B0",
    "DE": "#55A868",
    "DU": "#C44E52",
}

COLORS_2 = {
    "FC": "#8172B3",
    "MC": "#DD8452",
}

XGRID = np.linspace(-1, 1, 400)
FIG_WIDTH = 8.2
ROW_HEIGHT = 0.33
BW_METHOD = 0.45
# e.g. BW_METHOD = 0.8 for a bit less smoothing


# ----------------------------
# helpers
# ----------------------------

def pretty(name: str) -> str:
    return MODEL_NAME_MAP.get(name, name)


def normalize_model_name(name: str) -> str:
    name = name.replace("__", "_")
    if name.endswith("_fixed"):
        name = name[:-6]
    return name


def kde_curve(values: np.ndarray, bw_method=None):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return None

    if np.ptp(values) < 1e-10 or np.unique(values).size <= 1:
        return None

    try:
        kde = gaussian_kde(values, bw_method=bw_method)
    except np.linalg.LinAlgError:
        rng = np.random.default_rng(0)
        values = values + rng.normal(0, 1e-6, size=len(values))
        try:
            kde = gaussian_kde(values, bw_method=bw_method)
        except np.linalg.LinAlgError:
            return None

    y = kde(XGRID)
    if not np.all(np.isfinite(y)) or np.max(y) <= 0:
        return None

    return XGRID, y


def row_ymax(analysis: dict, dims: list[str], bw_method=None):
    ymax = 0.0
    for dim in dims:
        curve = kde_curve(analysis["bootstrap"]["draws"][dim], bw_method=bw_method)
        if curve is not None:
            _, y = curve
            ymax = max(ymax, float(np.max(y)))
    return ymax if ymax > 0 else 1.0


def draw_distribution(ax, values, point, color, alpha=0.26, lw=1.2, bw_method=None):
    curve = kde_curve(values, bw_method=bw_method)

    if curve is None:
        ax.axvline(point, color=color, linewidth=2, alpha=0.95, zorder=2)
    else:
        x, y = curve
        ax.fill_between(x, 0, y, color=color, alpha=alpha, zorder=2, clip_on=True)
        ax.plot(x, y, color=color, linewidth=lw, zorder=3, clip_on=True)

    ax.axvline(point, color=color, linewidth=1.0, alpha=0.95, zorder=4, clip_on=True)


def style_panel(ax, ymax):
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.7, alpha=0.55, zorder=1)
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, ymax * 1.05)
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)


# ----------------------------
# load analyses
# ----------------------------

part1 = {}
for path in Path("results_FC_part1").glob("*.pkl"):
    a = analyze_part1(path, n_boot=N_BOOT)
    a["model"] = normalize_model_name(a["model"])
    part1[a["model"]] = a

part2 = {}
for path in Path("results_FC_part2").glob("*.pkl"):
    a = analyze_part2(path, n_boot=N_BOOT)
    a["model"] = normalize_model_name(a["model"])
    part2[a["model"]] = a

if not part1:
    raise ValueError("No part I .pkl files found in results_FC_part1")

models = sorted(part1.keys(), key=lambda m: part1[m]["dimension_scores"]["FW"], reverse=True)
n_rows = len(models)


# ----------------------------
# figure
# ----------------------------

fig, axes = plt.subplots(
    n_rows,
    2,
    figsize=(FIG_WIDTH, ROW_HEIGHT * n_rows + 0.8),
    sharex=True,
    squeeze=False,
)

for i, model in enumerate(models):
    ax1, ax2 = axes[i, 0], axes[i, 1]

    # ----- part I -----
    a1 = part1[model]
    ymax1 = row_ymax(a1, ["FW", "DE", "DU"], bw_method=BW_METHOD)

    for dim in ["FW", "DE", "DU"]:
        draw_distribution(
            ax1,
            a1["bootstrap"]["draws"][dim],
            a1["dimension_scores"][dim],
            COLORS_1[dim],
            alpha=0.24,
            lw=1.15,
            bw_method=BW_METHOD,
        )
    style_panel(ax1, ymax1)

    ax1.set_ylabel(
        pretty(model),
        rotation=0,
        ha="right",
        va="center",
        labelpad=34,
        fontsize=8.5,
    )

    # ----- part II -----
    if model in part2:
        a2 = part2[model]
        ymax2 = row_ymax(a2, ["FC", "MC"], bw_method=BW_METHOD)

        for dim in ["FC", "MC"]:
            draw_distribution(
                ax2,
                a2["bootstrap"]["draws"][dim],
                a2["dimension_scores"][dim],
                COLORS_2[dim],
                alpha=0.28,
                lw=1.2,
                bw_method=BW_METHOD,
            )
        style_panel(ax2, ymax2)
    else:
        style_panel(ax2, 1.0)

    if i != n_rows - 1:
        ax1.set_xticklabels([])
        ax2.set_xticklabels([])

# titles
axes[0, 0].set_title("Part I", fontsize=11, pad=6)
axes[0, 1].set_title("Part II", fontsize=11, pad=6)

# bottom ticks
for ax in axes[-1, :]:
    ax.set_xticks([-1, 0, 1])
    ax.set_xticklabels(["-1", "0", "1"])
    ax.tick_params(axis="x", labelsize=8)

fig.subplots_adjust(
    left=0.28,
    right=0.98,
    top=0.93,
    bottom=0.08,
    wspace=0.12,
    hspace=0.35,
)

part1_handles = [
    plt.Line2D([0], [0], color=COLORS_1["FW"], lw=2, label="FW"),
    plt.Line2D([0], [0], color=COLORS_1["DE"], lw=2, label="DE"),
    plt.Line2D([0], [0], color=COLORS_1["DU"], lw=2, label="DU"),
]
part2_handles = [
    plt.Line2D([0], [0], color=COLORS_2["FC"], lw=2, label="FC"),
    plt.Line2D([0], [0], color=COLORS_2["MC"], lw=2, label="MC"),
]

fig.legend(
    part1_handles, ["FW", "DE", "DU"],
    loc="upper center", bbox_to_anchor=(0.33, 0.995),
    ncol=3, frameon=False, fontsize=8
)
fig.legend(
    part2_handles, ["FC", "MC"],
    loc="upper center", bbox_to_anchor=(0.78, 0.995),
    ncol=2, frameon=False, fontsize=8
)

fig.supxlabel("Dimension score", fontsize=10)

plt.savefig(OUTFILE, dpi=300, bbox_inches="tight")
plt.show()