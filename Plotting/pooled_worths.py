import matplotlib.pyplot as plt
import numpy as np

COLORS_1 = {
    "FW": "#4C72B0",
    "DE": "#55A868",
    "DU": "#C44E52",
}

COLORS_2 = {
    "FC": "#8172B3",
    "MC": "#DD8452",
}

# -----------------------------
# Data
# -----------------------------
part1_labels = ["FW1","FW2","FW3","FW4","FW5","DE1","DE2","DE3","DE4","DE5","DU1","DU2","DU3","DU4","DU5"]
part1_est = np.array([0.0000000,-0.6608669,0.1066795,0.1830379,0.4212475,2.2919255,3.3238872,2.7748482,3.6522574,2.3236788,-2.0667521,-0.9912497,0.7229732,0.4065224,-0.8185607])
part1_qse = np.array([0.2011383,0.1642160,0.1386643,0.1769294,0.1626822,0.1878459,0.3099240,0.2200281,0.3030781,0.1661599,0.2370140,0.1694440,0.1338182,0.1571842,0.1856110])

part1_worth = np.array([0.008785435,0.004560131,0.009832440,0.010606536,0.013444847,0.087862061,0.247677443,0.142835717,0.344017281,0.090595364,0.001109724,0.003269273,0.018250706,0.013278233,0.003874809])
part1_lower = np.array([0.0045670527,0.0023279550,0.0052542661,0.0056440580,0.0077061685,0.0445826927,0.1249817060,0.0710173037,0.1985792311,0.0479145736,0.0004841955,0.0017576197,0.0100305758,0.0068798438,0.0020604673])
part1_upper = np.array([0.012797417,0.006977535,0.014424812,0.015614945,0.017949743,0.139359254,0.432988518,0.244742454,0.565059851,0.140665854,0.001771597,0.004637822,0.026536031,0.019317234,0.005625527])

part2_labels = ["FC1","FC2","FC3","FC4","FC5","FC6","FC7","MC1","MC2","MC3","MC4","MC5","MC6","MC7"]
part2_est = np.array([0.000000000,2.563959032,1.492991173,0.873698596,0.572673592,0.047931173,0.094200833,0.623212485,0.326517720,-0.004930464,1.322865563,0.444449206,1.430357309,-0.214676525])
part2_qse = np.array([0.3477590,0.4059282,0.3561379,0.4646092,0.2995828,0.3079949,0.4704248,0.3657263,0.3831253,0.3219266,0.4709076,0.4642340,0.3374289,0.2935341])

part2_worth = np.array([0.02544460,0.33045239,0.11323833,0.06095904,0.04511330,0.02669389,0.02795803,0.04745188,0.03526960,0.02531946,0.09552320,0.03968419,0.10636333,0.02052876])
part2_lower = np.array([0.01299772,0.20319200,0.06409621,0.03938713,0.02456336,0.01355903,0.01720304,0.02488290,0.01809601,0.01314346,0.06296459,0.02508623,0.05643747,0.01059578])
part2_upper = np.array([0.04194058,0.48247414,0.18753196,0.08597478,0.07654971,0.04455676,0.03838084,0.07873573,0.05524782,0.04193599,0.13378484,0.05612722,0.18528916,0.03466293])

# -----------------------------
# Helpers
# -----------------------------
def prefix_color(label, color_map):
    prefix = ''.join([c for c in label if c.isalpha()])
    return color_map[prefix]

def make_side_by_side(labels, est, qse, worth, lower, upper, color_map, part_title):
    x = np.arange(len(labels))
    colors = [prefix_color(lbl, color_map) for lbl in labels]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

    # Left: quasiSE plot on log-worth scale
    ax = axes[0]
    for i, (xi, yi, err, c) in enumerate(zip(x, est, 2*qse, colors)):
        ax.errorbar(xi, yi, yerr=err, fmt='o', color=c, capsize=4)
    ax.axhline(0, linestyle='--', linewidth=1, color='black')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel("Log-worth estimate")
    ax.set_title(f"{part_title}: quasi-intervals (±2 quasiSE)")
    ax.grid(True, axis='y', alpha=0.3)

    # Right: bootstrap worth intervals
    ax = axes[1]
    for i, (xi, yi, lo, up, c) in enumerate(zip(x, worth, lower, upper, colors)):
        ax.errorbar(
            xi, yi,
            yerr=np.array([[yi - lo], [up - yi]]),
            fmt='o',
            color=c,
            capsize=4
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel("Worth")
    ax.set_title(f"{part_title}: bootstrap intervals")
    ax.grid(True, axis='y', alpha=0.3)

    fig.tight_layout()
    return fig, axes

# -----------------------------
# Make plots
# -----------------------------
fig1, axes1 = make_side_by_side(
    labels=part1_labels,
    est=part1_est,
    qse=part1_qse,
    worth=part1_worth,
    lower=part1_lower,
    upper=part1_upper,
    color_map=COLORS_1,
    part_title="Part One"
)

fig2, axes2 = make_side_by_side(
    labels=part2_labels,
    est=part2_est,
    qse=part2_qse,
    worth=part2_worth,
    lower=part2_lower,
    upper=part2_upper,
    color_map=COLORS_2,
    part_title="Part Two"
)

plt.show()