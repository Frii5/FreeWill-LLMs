
import numpy as np
from ridgeplot import ridgeplot
import plotly.graph_objects as go
from matplotlib import cm, colors as mcolors

from pathlib import Path
import pandas as pd

# Folder containing this script
SCRIPT_DIR = Path(__file__).resolve().parent
# Load data from the same folder as the script
boot = pd.read_csv(SCRIPT_DIR / "ridge_boot_part2.csv")
points = pd.read_csv(SCRIPT_DIR / "ridge_points_part2.csv")

# Set constants
METRIC = "Mean-item contrast"
CMAP_NAME = "autumn"
OUTPUT_FILE = SCRIPT_DIR / "ridge_mean_item_contrast.svg"

# Pretty display names
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
    "OVERALL": "Total"
}

# Intelligence scores
aa_scores = {
    "claude-haiku-4-5": 31,
    "claude-opus-4-6": 46,
    "claude-sonnet-4-6": 43,
    "deepseek-chat": 32,
    "gemini-3-flash-preview": 46,
    "gemini-3.1-pro-preview": 57,
    "google_gemma-3-4b": 6,
    "gpt-4o-2024-08-06": 19,
    "gpt-5.4-mini": 23,
    "gpt-5.4-nano": 24,
    "gpt-5.4": 35,
    "grok-4-1-fast-reasoning": 39,
    "grok-4.20-0309-reasoning": 49,
    "meta-llama-3-8b-instruct": 6,
    "mistral-large-2512": 23,
    "mistral-medium-2508": 21,
    "nvidia_nemotron-3-nano": 24,
    "openai_gpt-oss-20b": 21,
    "phi-4-reasoning-vision-15b": 10,
    "qwen_qwen3-vl-8b": 14,
    "OVERALL": np.nan
}

# Convert matplotlib colormap to Plotly rgba colors
def make_color_map(model_order, score_map, cmap_name=CMAP_NAME, alpha=0.7):
    valid_scores = np.array([
        score_map[m] for m in model_order
        if m in score_map and not np.isnan(score_map[m])
    ])

    norm = mcolors.Normalize(vmin=valid_scores.min(), vmax=valid_scores.max())
    cmap = cm.get_cmap(cmap_name)

    color_map = {}
    for model in model_order:
        label = pretty_names.get(model, model)
        score = score_map.get(model, np.nan)

        if np.isnan(score):
            color_map[label] = f"rgba(120,120,120,{alpha})"
        else:
            r, g, b, _ = cmap(norm(score))
            color_map[label] = f"rgba({int(255*r)},{int(255*g)},{int(255*b)},{alpha})"

    return color_map

# Convert matplotlib colormap to Plotly colorscale
def make_colorscale(cmap_name=CMAP_NAME, n=256):
    cmap = cm.get_cmap(cmap_name, n)

    return [
        [
            i / (n - 1),
            f"rgb({int(255*r)},{int(255*g)},{int(255*b)})"
        ]
        for i in range(n)
        for r, g, b, _ in [cmap(i / (n - 1))]
    ]

# Build ridgeplot data
boot_metric = boot.loc[boot["metric"] == METRIC].copy()
points_metric = points.loc[points["metric"] == METRIC].drop_duplicates("model").copy()

# Order models by original estimate
model_order = ["OVERALL"] + (
    points_metric.loc[points_metric["model"] != "OVERALL"]
    .sort_values("estimate", ascending=False)["model"]
    .tolist()
)

# Convert model IDs to display names
display_order = [pretty_names.get(model, model) for model in model_order]
display_lookup = dict(zip(model_order, display_order))

# Extract bootstrap samples per model
samples = [
    boot_metric.loc[boot_metric["model"] == model, "value"].dropna().to_numpy()
    for model in model_order
]

# Set KDE support
xmin, xmax = boot_metric["value"].min(), boot_metric["value"].max()
pad = 0.08 * (xmax - xmin)
kde_points = np.linspace(xmin - pad, xmax + pad, 500)

# Create ridgeplot
fig = ridgeplot(
    samples=samples,
    labels=display_order,
    row_labels=display_order,
    kde_points=kde_points,
    bandwidth="normal_reference",
    opacity=1,
    line_color="black",
    line_width=0.9,
    spacing=0.20,
    color_discrete_map=make_color_map(model_order, aa_scores)
)

# Add original estimates as points
points_metric["display_model"] = points_metric["model"].map(display_lookup)

fig.add_trace(
    go.Scatter(
        x=points_metric["estimate"],
        y=points_metric["display_model"],
        mode="markers",
        marker=dict(size=7, color="black"),
        hoverinfo="skip",
        showlegend=False
    )
)

# Add zero reference line
fig.add_vline(
    x=0,
    line_dash="4px,4px",
    line_color="rgba(0,0,0,0.35)",
    line_width=1
)

# Add colorbar
valid_scores = np.array([
    aa_scores[m] for m in model_order
    if m in aa_scores and not np.isnan(aa_scores[m])
])

fig.add_trace(
    go.Scatter(
        x=[None],
        y=[None],
        mode="markers",
        marker=dict(
            size=0.1,
            color=[valid_scores.min(), valid_scores.max()],
            cmin=valid_scores.min(),
            cmax=valid_scores.max(),
            colorscale=make_colorscale(),
            showscale=True,
            colorbar=dict(
                title=dict(text="Intelligence Index", side="right"),
                thickness=12,
                len=0.765,
                y=0.412,
                yanchor="middle",
                x=0.98
            )
        ),
        hoverinfo="skip",
        showlegend=False
    )
)

# Style layout
fig.update_layout(
    xaxis_title="Bootstrapped Worth Contrast",
    yaxis_title=None,
    width=550,
    height=max(650, 28 * len(model_order)),
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(size=12, family="Times New Roman", color="black"),
    showlegend=False,
    margin=dict(r=90)
)

# Style x-axis
fig.update_xaxes(
    showgrid=True,
    gridcolor="rgba(0,0,0,0.08)",
    gridwidth=1,
    zeroline=False,
    showline=True,
    linecolor="rgba(0,0,0,0.25)",
    linewidth=1,
    ticks="outside",
    tickcolor="rgba(0,0,0,0.35)",
    tickwidth=1,
    ticklen=4
)

# Style y-axis
fig.update_yaxes(
    showgrid=False,
    showline=False,
    ticks="",
    tickfont=dict(size=11),
    categoryorder="array",
    categoryarray=display_order[::-1]
)

# Show and save figure
fig.show()
fig.write_image(OUTPUT_FILE)