"""Step 3: charts.

    ./venv/bin/python charts.py   -> charts/01_years.png, 02_decomposition.png, 03_attributes.png, 04_physics.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
LAST_X = 2025
INK, DIM, GRID, BG, BLUE, ORANGE, GRAY = "#f2f2f0", "#8a8a87", "#1d1d1d", "#0b0b0b", "#3987e5", "#d95926", "#9a9a96"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": INK,
    "axes.edgecolor": "#3a3a3a", "axes.labelcolor": DIM, "xtick.color": DIM, "ytick.color": DIM,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 1,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 11, "axes.titlesize": 13,
    "axes.titlelocation": "left", "axes.titlepad": 12,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(HERE / "charts" / name, dpi=150)
    plt.close(fig)


def main():
    (HERE / "charts").mkdir(exist_ok=True)
    ye = pd.read_csv(HERE / "results" / "year_effects.csv")
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ends = []
    for label, color in [("raw average", GRAY), ("same weight, power, body, drive", BLUE), ("...and same engine and transmission", ORANGE)]:
        s = ye[ye["model"] == label]
        ax.fill_between(s["year"], s["lo"], s["hi"], color=color, alpha=0.12, linewidth=0)
        ax.plot(s["year"], s["pct"], color=color, linewidth=2, marker="o", markersize=4, label=label)
        ends.append(s["pct"].iloc[-1])
    ys = sorted(ends)                                                  # nudge end labels apart so close values don't overlap
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + 0.9)
    for v in ends:
        ax.text(LAST_X + 0.15, ys[sorted(ends).index(v)], f"{v:+.1f}%", color=INK, va="center", fontsize=10)
    ax.axhline(0, color=DIM, linewidth=1)
    ax.set_xlim(2012.7, 2026.1)
    ax.set_xticks(range(2013, 2026, 2))
    ax.set_ylabel(f"fuel used vs 2013 (%)")
    ax.legend(frameon=False, loc="lower left", labelcolor=INK)
    ax.set_title("At the same weight and power, 2025 cars use 14% less fuel than 2013 ones (95% CI shaded)")
    save(fig, "01_years.png")

    d = json.loads((HERE / "results" / "decomposition.json").read_text())
    parts = d["parts_pct"]
    labels = ["technology\n(same size and power)", "heavier", "more powerful", "body and drive mix", "net change"]
    vals = [parts["technology (year effect)"], parts["heavier"], parts["more powerful"], parts["body and drive mix"], d["actual_pct"]]
    logs = [d["parts_log"]["technology (year effect)"], d["parts_log"]["heavier"], d["parts_log"]["more powerful"], d["parts_log"]["body and drive mix"]]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    start = 0.0
    for i, (lab, v) in enumerate(zip(labels, vals)):
        if i < 4:                                                     # bars stack in log space, so the pieces add up exactly
            left, width = start, logs[i]
            start += logs[i]
        else:
            left, width = 0.0, d["actual_log_change"]
        color = GRAY if i == 4 else (BLUE if v < 0 else ORANGE)
        ax.barh(i, 100 * width, left=100 * left, color=color, height=0.55)
        x = 100 * (left + width)
        ax.text(x + (0.3 if width > 0 else -0.3), i, f"{v:+.1f}%", va="center", ha="left" if width > 0 else "right", color=INK, fontsize=10)
    ax.set_yticks(range(5), labels)
    ax.invert_yaxis()
    ax.axvline(0, color=DIM, linewidth=1)
    ax.set_xlim(-19, 3)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("change in fuel used per mile, 2013 to 2025 (bars add up in log points x 100; labels are % changes)")
    ax.set_title(f"Bigger, more powerful cars took back {100 * d['share_of_technology_taken_back']:.0f}% of the technology gain")
    save(fig, "02_decomposition.png")

    f = pd.read_csv(HERE / "results" / "model_data.csv")
    g = f.groupby("year")[["weight", "hp"]].mean()
    g = 100 * g / g.iloc[0]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for col, color, name in [("hp", ORANGE, "horsepower"), ("weight", BLUE, "test weight")]:
        ax.plot(g.index, g[col], color=color, linewidth=2, marker="o", markersize=4, label=name)
        ax.text(g.index[-1] + 0.15, g[col].iloc[-1], f"{name} {g[col].iloc[-1]:.0f}", color=INK, va="center", fontsize=10)
    ax.axhline(100, color=DIM, linewidth=1)
    ax.set_xlim(2012.7, 2026.9)
    ax.set_xticks(range(2013, 2026, 2))
    ax.set_ylabel("average, 2013 = 100")
    ax.legend(frameon=False, loc="upper left", labelcolor=INK)
    ax.set_title("Tested cars got heavier and more powerful (average per year, 2013 = 100)")
    save(fig, "03_attributes.png")

    p = pd.read_csv(HERE / "results" / "physics.csv")
    names = {"log_weight": "test weight", "log_hp": "horsepower", "log_c": "air drag coefficient", "log_a": "rolling/friction coefficient"}
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for k, (cycle, color, dy) in enumerate([("city", BLUE, -0.13), ("highway", ORANGE, 0.13)]):
        s = p[p["cycle"] == cycle].set_index("term").loc[list(names)]
        y = np.arange(len(names)) + dy
        ax.hlines(y, s["lo"], s["hi"], color=color, linewidth=2, alpha=0.6)
        ax.scatter(s["elasticity"], y, color=color, s=45, zorder=3, label=f"{cycle} test")
    ax.set_yticks(range(len(names)), list(names.values()))
    ax.invert_yaxis()
    ax.axvline(0, color=DIM, linewidth=1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("% more fuel per 1% more of each (elasticity, 95% CI, clustered by manufacturer)")
    ax.legend(frameon=False, loc="lower right", labelcolor=INK)
    ax.set_title("Air drag matters about twice as much on the highway as in the city")
    save(fig, "04_physics.png")
    print("charts written")


if __name__ == "__main__":
    main()
