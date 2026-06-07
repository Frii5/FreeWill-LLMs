import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

SAVE_FIGURES = True

plt.rcParams["font.family"] = "Times New Roman"

CELL_WIDTH = 0.35
CELL_HEIGHT = 0.20

TITLE_SIZE = 12
TICK_SIZE = 12
ANNOT_SIZE = 9
COLORBAR_TICK_SIZE = 9

TEXT_FONT = "Times New Roman"
NUMBER_FONT = "DejaVu Sans"

COLORBAR_PAD = 0.015

COMPAT_COLOR = "#4640FF"     
INCOMPAT_COLOR = "#000000"   
NEUTRAL_COLOR = "#5F5F5F"    

df_part1 = pd.read_csv("worths_by_model_part1.csv").set_index("model")
df_part2 = pd.read_csv("worths_by_model_part2.csv").set_index("model")

pretty_names = {
    "meta-llama-3-8b-instruct": "Llama 3 8B",
    "mistral-large-2512": "Mistral Large",
    "gpt-4o-2024-08-06": "GPT-4o",
    "phi-4-reasoning-vision-15b": "Phi-4 RV 15B",
    "claude-haiku-4-5": "Claude Haiku 4.5",
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "deepseek-chat": "DeepSeek Chat",
    "gemini-3-flash-preview": "Gemini 3 Flash",
    "google_gemma-3-4b": "Gemma 3 4B",
    "gpt-5.4-mini": "GPT-5.4 Mini",
    "grok-4.20-0309-reasoning": "Grok 4.20",
    "gpt-5.4-nano": "GPT-5.4 Nano",
    "grok-4-1-fast-reasoning": "Grok 4.1 Fast",
    "qwen_qwen3-vl-8b": "Qwen3-VL 8B",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "openai_gpt-oss-20b": "GPT-OSS 20B",
    "claude-opus-4-6": "Claude Opus 4.6",
    "nvidia_nemotron-3-nano": "Nemotron 3 Nano",
    "gpt-5.4": "GPT-5.4",
    "mistral-medium-2508": "Mistral Medium",
}

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

item_label_colors_part2 = {
    "FC1": INCOMPAT_COLOR,
    "FC4": INCOMPAT_COLOR,
    "FC6": INCOMPAT_COLOR,
    "MC1": INCOMPAT_COLOR,
    "MC3": INCOMPAT_COLOR,
    "FC2": COMPAT_COLOR,
    "FC3": COMPAT_COLOR,
    "FC7": COMPAT_COLOR,
    "MC4": COMPAT_COLOR,
    "MC5": COMPAT_COLOR,
    "MC6": COMPAT_COLOR,
    "FC5": NEUTRAL_COLOR,
    "MC2": NEUTRAL_COLOR,
    "MC7": NEUTRAL_COLOR,
}

# compatibilist contrasts
# Used only for ordering rows
compat_scores = {
    "claude-haiku-4-5": 0.039715,
    "claude-opus-4-6": 0.071025,
    "claude-sonnet-4-6": 0.073242,
    "deepseek-chat": 0.060424,
    "gemini-3-flash-preview": 0.062934,
    "gemini-3.1-pro-preview": 0.030350,
    "google_gemma-3-4b": -0.041700,
    "gpt-4o-2024-08-06": 0.017341,
    "gpt-5.4-mini": 0.068778,
    "gpt-5.4-nano": 0.004474,
    "gpt-5.4": 0.066778,
    "grok-4-1-fast-reasoning": 0.055161,
    "grok-4.20-0309-reasoning": 0.052757,
    "meta-llama-3-8b-instruct": -0.028500,
    "mistral-large-2512": 0.008998,
    "mistral-medium-2508": 0.068406,
    "nvidia_nemotron-3-nano": 0.015606,
    "openai_gpt-oss-20b": 0.023434,
    "phi-4-reasoning-vision-15b": 0.029142,
    "qwen_qwen3-vl-8b": 0.012672,
}

model_order = sorted(
    compat_scores,
    key=compat_scores.get,
    reverse=True
)

# Apply order and pretty names
df_part1 = df_part1.loc[model_order].rename(index=pretty_names)
df_part2 = df_part2.loc[model_order].rename(index=pretty_names)

def plot_heatmap(
    df,
    output_stem,
    separators,
    annotate=True,
    item_label_colors=None
):
    n_rows, n_cols = df.shape

    vmin = df.min().min()
    vmax = df.max().max()

    heatmap_width = n_cols * CELL_WIDTH
    heatmap_height = n_rows * CELL_HEIGHT

    fig_width = heatmap_width + 4.0
    fig_height = heatmap_height + 1.4

    plt.figure(figsize=(fig_width, fig_height))

    ax = sns.heatmap(
        df,
        cmap="Blues",
        vmin=vmin,
        vmax=vmax,
        annot=annotate,
        fmt=".2f",
        annot_kws={
            "size": ANNOT_SIZE,
            "family": NUMBER_FONT
        },
        linewidths=0.25,
        linecolor="white",
        square=False,
        cbar_kws={
            "shrink": 0.85,
            "pad": COLORBAR_PAD
        }
    )

    for x in separators:
        ax.axvline(x, color="white", linewidth=2.0)

    ax.set_xlabel("")
    ax.set_ylabel("")

    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        fontsize=TICK_SIZE,
        fontfamily=TEXT_FONT
    )

    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=0,
        fontsize=TICK_SIZE,
        fontfamily=TEXT_FONT
    )

    if item_label_colors is not None:
        for label in ax.get_xticklabels():
            item = label.get_text()
            label.set_color(item_label_colors.get(item, NEUTRAL_COLOR))
            label.set_fontweight("bold")

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_SIZE)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    cbar.update_ticks()

    for tick in cbar.ax.get_yticklabels():
        tick.set_fontfamily(NUMBER_FONT)
        tick.set_fontsize(COLORBAR_TICK_SIZE)

    plt.tight_layout()

    if SAVE_FIGURES:
        #plt.savefig(f"{output_stem}.pdf", bbox_inches="tight")
        plt.savefig(f"Plotting/{output_stem}.svg", bbox_inches="tight")

    plt.show()

plot_heatmap(
    df=df_part2,
    output_stem="heatmap_part2",
    separators=[7],
    annotate=True,
    item_label_colors=item_label_colors_part2
)

plot_heatmap(
    df=df_part1,
    output_stem="heatmap_part1",
    separators=[5, 10],
    annotate=True,
    item_label_colors=None
)