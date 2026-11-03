"""Draw the paper's Figure 1 (fig_main.pdf) from the three 2012 test summaries.

Run from control/ with its environment:

    uv run python ../paper/figures/make_figure.py

It reads control/results/analysis/<split>/summary.csv for amsterdam_2012,
london_2012 and abbotsford_2012, writes paper/figures/fig_main.pdf at the
column width (3.3 in, so its 9 pt text prints at size), and prints every
value it plots so the figure can be checked against the files.
"""

from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "fig_main.pdf"
W, H = 3.3, 4.2  # inches
SPLITS = {"amsterdam_2012": "Amsterdam", "london_2012": "London", "abbotsford_2012": "Abbotsford"}
# Study label -> display name and marker, in legend order.
CONTROLLERS = {
    "rb_default": ("Default rules", "s"),
    "rb_tuned": ("Tuned rules", "D"),
    "residual_w05": ("Residual (fixed penalty)", "^"),
    "lag_b100": ("Residual (constrained)", "v"),
    "ppo": ("PPO", "o"),
    "zero_controls": ("All off", "P"),
}
CITY_FILL = {"Amsterdam": "black", "London": "white", "Abbotsford": "0.6"}
MARK = {"ms": 5, "mec": "black", "mew": 0.6, "ls": "none"}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
    }
)


def rect(left, bottom, right, top):
    """An add_axes rectangle from edges in inches."""
    return [left / W, bottom / H, (right - left) / W, (top - bottom) / H]


def load():
    """Return {city: summary DataFrame indexed by label}."""
    return {
        city: pd.read_csv(ROOT / "control/results/analysis" / split / "summary.csv").set_index("label")
        for split, city in SPLITS.items()
    }


def draw_sources(ax, summaries):
    """Panel (a): PPO's lead over the default rules, split into tuning and learning."""
    print("(a) profit_gain_iqm: tuning = -rb_default, PPO = ppo, share = tuning / (tuning + PPO)")
    cities = list(SPLITS.values())
    for city in cities:
        y = len(cities) - 1 - cities.index(city)
        tune = -summaries[city].loc["rb_default", "profit_gain_iqm"]
        ppo = summaries[city].loc["ppo", "profit_gain_iqm"]
        share = tune / (tune + ppo)
        print(f"  {city:<10} tuning {tune:.4f}  PPO {ppo:.4f}  share {share:.4f} -> {share:.0%}")
        ax.barh(y, tune, color="0.85", edgecolor="black", linewidth=0.6, height=0.62)
        ax.barh(y, ppo, left=tune, color="0.3", edgecolor="black", linewidth=0.6, height=0.62)
        ax.text(tune / 2, y, f"{share:.0%}", ha="center", va="center")
    ax.set_yticks(range(len(cities)), list(reversed(cities)))
    ax.set_ylim(-0.5, len(cities) - 0.5)
    ax.set_xlim(0, 13)
    ax.set_xlabel("Profit gain (€/m² per 60-day season)")
    ax.tick_params(axis="y", length=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        ["Tuning the rules", "PPO over the tuned rules"],
        loc="lower right",
        bbox_to_anchor=(1.0, 1.0),
        ncol=2,
        frameon=False,
        borderaxespad=0.1,
        handlelength=1.2,
        columnspacing=1.0,
    )


def draw_objective(ax, summaries):
    """Panel (b): profit IQM against fruit growth IQM, one marker per controller and city."""
    print("(b) yield_kg_m2_iqm (x), profit_eur_m2_iqm (y)")
    for label, (name, marker) in CONTROLLERS.items():
        for city, fill in CITY_FILL.items():
            x = summaries[city].loc[label, "yield_kg_m2_iqm"]
            y = summaries[city].loc[label, "profit_eur_m2_iqm"]
            print(f"  {name:<25} {city:<10} fruit {x:.4f}  profit {y:.4f}")
            ax.plot(x, y, marker=marker, mfc=fill, **MARK)
    controller_key = [plt.Line2D([], [], marker=m, mfc="white", label=n, **MARK) for n, m in CONTROLLERS.values()]
    # Filled rectangles, not markers: every marker shape already names a controller.
    city_key = [plt.Rectangle((0, 0), 1, 1, fc=f, ec="black", lw=0.6, label=c) for c, f in CITY_FILL.items()]
    controller_legend = ax.legend(
        handles=controller_key,
        loc="lower left",
        frameon=False,
        borderaxespad=0.2,
        labelspacing=0.4,
        handletextpad=0.3,
    )
    ax.add_artist(controller_legend)
    ax.legend(
        handles=city_key,
        loc="lower left",
        bbox_to_anchor=(-0.02, 1.0),
        ncol=3,
        frameon=False,
        borderaxespad=0.1,
        handlelength=1.2,
        handletextpad=0.4,
        columnspacing=1.0,
    )
    ax.axhline(0, color="0.5", lw=0.5, ls=":", zorder=0)
    ax.set_xlim(0, 12.5)
    ax.set_xlabel("Fruit growth (kg/m²)")
    ax.set_ylabel("Profit (€/m²)", labelpad=2)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    summaries = load()
    fig = plt.figure(figsize=(W, H))
    ax_a = fig.add_axes(rect(0.68, 2.95, 3.22, 3.72))
    ax_b = fig.add_axes(rect(0.5, 0.42, 3.22, 2.0))
    draw_sources(ax_a, summaries)
    draw_objective(ax_b, summaries)
    fig.text(0.02 / W, 4.18 / H, "(a) Where PPO’s lead over the default rules comes from", va="top")
    fig.text(0.02 / W, 2.42 / H, "(b) What profit rewards", va="top")
    fig.savefig(OUT, metadata={"CreationDate": None})
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
