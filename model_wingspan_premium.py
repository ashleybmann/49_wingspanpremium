"""Model the 'wingspan premium': does excess length (wingspan - barefoot height)
buy a prospect an earlier draft slot?

Fits OLS draft_pick ~ wingspan_diff, prints the regression summary plus two
outlier tables, and saves wingspan_premium_scatter.png.

Restricted to the voluntary combine era (2015-2023): from 2024 on, combine
attendance is mandatory for draft-eligible prospects, which changes who is measured.
"""

import itertools
import math

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as smf
from matplotlib.colors import TwoSlopeNorm

INPUT_CSV = "nba_draft_combine_wingspan.csv"
OUTPUT_PNG = "wingspan_premium_scatter.png"
TOP_N = 5
ANNOTATE_N = 5
FIRST_ROUND_MAX_PICK = 30
LOTTERY_MAX_PICK = 14
# A "Length Reach" must actually be long: top quartile of wingspan_diff.
LENGTH_QUANTILE = 0.75
FIRST_YEAR, LAST_YEAR = 2015, 2023

TITLE = "The Wingspan Premium: Measuring NBA Front-Office Reach for Raw Physical Length"

INK = "#1f1f1e"
MUTED = "#6b6b68"
GRID = "#e6e6e3"
SURFACE = "#fcfcfb"


def load_data(path):
    df = pd.read_csv(path)
    df = df[df["draft_year"].between(FIRST_YEAR, LAST_YEAR)]
    return df.rename(columns={"overall_pick": "draft_pick"})


def fit_model(df):
    model = smf.ols("draft_pick ~ wingspan_diff", data=df).fit()
    df = df.assign(predicted_pick=model.fittedvalues, residual=model.resid)
    return model, df


def print_summary(model):
    b0, b1 = model.params["Intercept"], model.params["wingspan_diff"]
    p = model.pvalues["wingspan_diff"]
    lo, hi = model.conf_int().loc["wingspan_diff"]
    print("=" * 72)
    print("OLS: draft_pick = b0 + b1 * wingspan_diff")
    print("=" * 72)
    print(f"  Sample size (n)      : {int(model.nobs)}")
    print(f"  R-squared            : {model.rsquared:.5f}")
    print(f"  Intercept (b0)       : {b0:.3f}")
    print(f"  Slope (b1)           : {b1:+.3f} picks per inch  (95% CI {lo:+.2f} to {hi:+.2f})")
    print(f"  p-value (slope)      : {p:.4f}")
    direction = "earlier" if b1 < 0 else "later"
    print(f"\n  Each extra inch of wingspan_diff moves the predicted slot {abs(b1):.2f} picks {direction}.")
    if p >= 0.05:
        print("  The slope is not statistically distinguishable from zero: no measurable premium.")


def length_reaches(df):
    cutoff = df["wingspan_diff"].quantile(LENGTH_QUANTILE)
    long_players = df[df["wingspan_diff"] >= cutoff]
    return long_players.nsmallest(TOP_N, "residual"), cutoff


def skill_exceptions(df):
    first_round = df[df["draft_pick"] <= FIRST_ROUND_MAX_PICK]
    return first_round.sort_values(["wingspan_diff", "draft_pick"]).head(TOP_N)


def print_table(title, table, note):
    out = table.assign(
        tier=table["draft_pick"].map(lambda p: "Lottery" if p <= LOTTERY_MAX_PICK else "1st Rd")
    )[
        ["player_name", "draft_year", "team_abbreviation", "draft_pick", "tier",
         "wingspan_diff", "predicted_pick", "residual"]
    ].rename(columns={
        "player_name": "Player", "draft_year": "Year", "team_abbreviation": "Team",
        "draft_pick": "Pick", "tier": "Tier", "wingspan_diff": "Wingspan +/- (in)",
        "predicted_pick": "Predicted", "residual": "Residual",
    })
    out.index = range(1, len(out) + 1)
    print("\n" + "=" * 72)
    print(title)
    print(note)
    print("=" * 72)
    print(out.to_string(float_format=lambda v: f"{v:.2f}"))


def _segments_cross(p1, p2, p3, p4):
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    return (orient(p1, p2, p3) * orient(p1, p2, p4) < 0
            and orient(p3, p4, p1) * orient(p3, p4, p2) < 0)


def label_column(ax, rows, x_text, y_top, y_bottom, ha, color):
    """Stack labels in an evenly spaced column with leader lines, so they never collide.

    Label order is the permutation with the fewest leader-line crossings (then shortest total length).
    """
    n = len(rows)
    step = (y_bottom - y_top) / max(n - 1, 1)
    slots = [(x_text, y_top + i * step) for i in range(n)]
    points = list(zip(rows["wingspan_diff"], rows["draft_pick"]))

    def cost(order):
        segs = [(points[j], slots[i]) for i, j in enumerate(order)]
        crossings = sum(_segments_cross(*segs[i], *segs[k])
                        for i, k in itertools.combinations(range(n), 2))
        length = sum(math.dist(pt, slot) for pt, slot in segs)
        return crossings, length

    best = min(itertools.permutations(range(n)), key=cost)
    rows = rows.iloc[list(best)]
    ax.scatter(rows["wingspan_diff"], rows["draft_pick"], s=110, facecolors="none",
               edgecolors=color, linewidths=1.2, zorder=4)
    for i, (_, r) in enumerate(rows.iterrows()):
        ax.annotate(
            f"{r['player_name']} ('{str(r['draft_year'])[-2:]})  #{r['draft_pick']}",
            xy=(r["wingspan_diff"], r["draft_pick"]),
            xytext=(x_text, y_top + i * step),
            ha=ha, va="center", fontsize=8.5, color=INK,
            arrowprops=dict(arrowstyle="-", color=color, lw=0.9, shrinkA=2, shrinkB=4),
            bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE, ec=color, lw=0.8),
            zorder=5,
        )


def plot(df, model, reaches, exceptions):
    b0, b1 = model.params["Intercept"], model.params["wingspan_diff"]
    p = model.pvalues["wingspan_diff"]

    fig, ax = plt.subplots(figsize=(13, 8.5), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    lim = df["residual"].abs().max()
    sc = ax.scatter(
        df["wingspan_diff"], df["draft_pick"],
        c=df["residual"], cmap="coolwarm_r", norm=TwoSlopeNorm(0, -lim, lim),
        s=42, edgecolors=SURFACE, linewidths=0.8, alpha=0.95, zorder=3,
    )

    x_min, x_max = df["wingspan_diff"].min() - 0.5, df["wingspan_diff"].max() + 0.5
    xs = pd.Series([x_min, x_max])
    ax.plot(xs, b0 + b1 * xs, color=INK, lw=2, zorder=4)
    ax.text(
        x_max - 0.1, b0 + b1 * x_max + 1.0,
        f"OLS fit: slope = {b1:+.2f} picks/in, p = {p:.2f}  (R² = {model.rsquared:.4f})",
        ha="right", va="top", fontsize=9, color=INK, zorder=6,
        bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE, ec="none", alpha=0.9),
    )

    ax.axhline(FIRST_ROUND_MAX_PICK + 0.5, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.text(x_min + 0.1, FIRST_ROUND_MAX_PICK + 0.2, "End of Round 1",
            fontsize=8, color=MUTED, va="bottom")

    # Labels live in the empty bands on either side of the cloud: reaches on the right
    # (long arms), skill exceptions on the left (short arms).
    label_column(ax, reaches.head(ANNOTATE_N), x_max + 0.6, 2, 22, "left", INK)
    label_column(ax, exceptions.head(ANNOTATE_N), x_min - 0.6, 6, 26, "right", INK)

    ax.set_xlim(x_min - 4.8, x_max + 5.2)
    ax.set_ylim(61, 0)  # inverted: Pick 1 at the top
    ax.set_yticks([1, 10, 20, 30, 40, 50, 60])
    ax.set_xlabel("Wingspan − Barefoot Height (inches)", fontsize=11, color=INK)
    ax.set_ylabel("Draft Pick (1 = first overall)", fontsize=11, color=INK)
    ax.set_title(TITLE, fontsize=14, fontweight="bold", color=INK, loc="left", pad=28)
    ax.text(
        0, 1.015,
        f"NBA Draft Combine participants who were drafted, {FIRST_YEAR}–{LAST_YEAR} (Voluntary Era), "
        f"n = {int(model.nobs)}. "
        f"Labeled: top {ANNOTATE_N} Length Reaches (right) and Skill/Craft Exceptions (left).",
        transform=ax.transAxes, fontsize=9.5, color=MUTED,
    )

    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)

    cbar = fig.colorbar(sc, ax=ax, pad=0.01, fraction=0.03)
    cbar.set_label("Residual (picks)  ←  drafted earlier than predicted  |  later  →",
                   fontsize=9, color=INK)
    cbar.ax.tick_params(colors=MUTED, labelsize=8)
    cbar.outline.set_visible(False)

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main():
    df = load_data(INPUT_CSV)
    model, df = fit_model(df)
    print_summary(model)

    reaches, cutoff = length_reaches(df)
    print_table(
        f"TOP {TOP_N} LENGTH REACHES",
        reaches,
        f"Players with wingspan_diff >= {cutoff:.2f} in (top quartile), "
        "sorted by most negative residual (drafted earliest vs. the line).",
    )

    exceptions = skill_exceptions(df)
    print_table(
        f"TOP {TOP_N} SKILL / CRAFT EXCEPTIONS",
        exceptions,
        "First-round picks sorted by lowest wingspan_diff (ties: earlier pick first).",
    )

    plot(df, model, reaches, exceptions)
    print(f"\nSaved chart -> {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
