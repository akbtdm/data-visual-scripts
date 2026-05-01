"""
Task 4: Data Visualization — 5 Charts for the Almaty Housing Market Analysis
Outputs: PNG files + an interactive HTML Plotly dashboard.
"""

import logging
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import plotly.express as px

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PALETTE = "husl"
FIGURE_DPI = 150
OUTPUT_DIR = Path("charts")
SEGMENT_NEW = "Новостройка"
SEGMENT_SEC = "Вторичка"


def _setup_output_dir():
    OUTPUT_DIR.mkdir(exist_ok=True)


def _with_segment_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Is_NewBuild" not in out.columns:
        out["Is_NewBuild"] = False
    out["Is_NewBuild"] = out["Is_NewBuild"].fillna(False).astype(bool)
    out["Segment"] = np.where(out["Is_NewBuild"], SEGMENT_NEW, SEGMENT_SEC)
    return out


def _fmt_millions(x, _):
    """Format axis ticks as 'N млн' for readability."""
    if x >= 1_000_000:
        return f"{x/1_000_000:.0f}M"
    if x >= 1_000:
        return f"{x/1_000:.0f}K"
    return str(int(x))


def _fmt_price_per_m2_axis(x, _pos=None):
    """Axis labels for ₸/m² (avoids duplicate 'M' ticks from _fmt_millions)."""
    xa = float(x)
    if abs(xa) >= 1_000_000:
        return f"{xa/1_000_000:.1f}M"
    if abs(xa) >= 1_000:
        return f"{xa/1_000:.0f}K"
    return f"{xa:.0f}"


def _newbuild_counts_line(df: pd.DataFrame) -> str:
    d = _with_segment_column(df)
    n_new = int(d["Is_NewBuild"].sum())
    n_sec = int(len(d) - n_new)
    return f"Новостройка: n={n_new}  |  Вторичка: n={n_sec}"


# ── Chart 1: Box Plot — Price distribution by District ─────────────────────
def chart_boxplot_price_by_district(df: pd.DataFrame):
    order = (
        df.groupby("District")["Price_KZT"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.boxplot(
        data=df,
        x="District",
        y="Price_KZT",
        order=order,
        palette=PALETTE,
        showfliers=False,
        ax=ax,
    )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_millions))
    ax.set_title("Price Distribution by Almaty District", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("District", fontsize=12)
    ax.set_ylabel("Price (KZT)", fontsize=12)
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.tight_layout()
    path = OUTPUT_DIR / "1_boxplot_price_by_district.png"
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info("Saved %s", path)


# ── Chart 2: Scatter Plot — Area vs Price, colored by Room Count ────────────
def chart_scatter_area_vs_price(df: pd.DataFrame):
    plot_df = df.dropna(subset=["Area_m2", "Price_KZT", "Rooms"]).copy()
    plot_df["Rooms_str"] = plot_df["Rooms"].astype(int).astype(str) + "-комн."

    fig = px.scatter(
        plot_df,
        x="Area_m2",
        y="Price_KZT",
        color="Rooms_str",
        hover_data=["District", "Address", "Floor_Info"],
        title="Area vs Price — Colored by Room Count",
        labels={"Area_m2": "Area (m²)", "Price_KZT": "Price (KZT)", "Rooms_str": "Rooms"},
        template="plotly_white",
        opacity=0.65,
    )
    fig.update_layout(
        title_font_size=18,
        xaxis_title_font_size=13,
        yaxis_title_font_size=13,
        legend_title_font_size=12,
    )
    path = OUTPUT_DIR / "2_scatter_area_vs_price.html"
    fig.write_html(str(path))
    logger.info("Saved %s", path)

    # Also save static PNG
    static_fig, ax = plt.subplots(figsize=(12, 7))
    rooms_sorted = sorted(plot_df["Rooms"].unique())
    colors = sns.color_palette(PALETTE, len(rooms_sorted))
    for room, color in zip(rooms_sorted, colors):
        subset = plot_df[plot_df["Rooms"] == room]
        ax.scatter(
            subset["Area_m2"], subset["Price_KZT"],
            label=f"{int(room)}-комн.",
            alpha=0.55, s=25, color=color,
        )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_millions))
    ax.set_title("Area vs Price (colored by Room Count)", fontsize=15, fontweight="bold")
    ax.set_xlabel("Area (m²)", fontsize=12)
    ax.set_ylabel("Price (KZT)", fontsize=12)
    ax.legend(title="Rooms", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    png_path = OUTPUT_DIR / "2_scatter_area_vs_price.png"
    static_fig.savefig(png_path, dpi=FIGURE_DPI)
    plt.close(static_fig)
    logger.info("Saved %s", png_path)


# ── Chart 3: Heatmap — Correlation Matrix ───────────────────────────────────
def chart_heatmap_correlation(df: pd.DataFrame):
    cols = ["Price_KZT", "Area_m2", "Rooms", "Current_Floor", "Total_Floors", "Price_per_m2"]
    corr_df = df[cols].dropna().corr().round(2)

    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr_df, dtype=bool), k=1)
    sns.heatmap(
        corr_df,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        ax=ax,
        annot_kws={"size": 11},
    )
    ax.set_title(
        "Correlation Heatmap: Price, Area, Rooms, Floor",
        fontsize=14, fontweight="bold", pad=12,
    )
    plt.tight_layout()
    path = OUTPUT_DIR / "3_heatmap_correlation.png"
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info("Saved %s", path)


# ── Chart 4: Bar Chart — Avg Price per m2 by District (sorted) ─────────────
def chart_bar_avg_price_per_m2_by_district(df: pd.DataFrame):
    agg = (
        df.groupby("District")["Price_per_m2"]
        .mean()
        .sort_values(ascending=True)
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.barh(
        agg["District"],
        agg["Price_per_m2"],
        color=sns.color_palette(PALETTE, len(agg)),
        edgecolor="white",
        linewidth=0.5,
    )
    for bar, val in zip(bars, agg["Price_per_m2"]):
        ax.text(
            bar.get_width() + 50,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,.0f} ₸/m²",
            va="center", fontsize=9,
        )
    ax.set_title("Average Price per m² by District (Sorted)", fontsize=15, fontweight="bold")
    ax.set_xlabel("Price per m² (KZT)", fontsize=12)
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_millions))
    plt.tight_layout()
    path = OUTPUT_DIR / "4_bar_avg_price_per_m2_district.png"
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info("Saved %s", path)


# ── Chart 5: Histogram — Distribution of Price_per_m2 ─────────────────────
def chart_histogram_price_per_m2(df: pd.DataFrame):
    data = df["Price_per_m2"].dropna()

    fig, ax = plt.subplots(figsize=(12, 6))
    n, bins, patches = ax.hist(
        data, bins=60, color="#4C9BE8", edgecolor="white", linewidth=0.4, alpha=0.85
    )
    # Shade the "sweet spot" (25th–75th percentile)
    p25, p75 = data.quantile(0.25), data.quantile(0.75)
    for patch, left in zip(patches, bins[:-1]):
        if p25 <= left <= p75:
            patch.set_facecolor("#E8844C")
            patch.set_alpha(0.9)

    ax.axvline(data.median(), color="#d62728", linestyle="--", linewidth=1.8,
               label=f"Median: {data.median():,.0f} ₸/m²")
    ax.axvline(data.mean(), color="#2ca02c", linestyle="--", linewidth=1.8,
               label=f"Mean: {data.mean():,.0f} ₸/m²")

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_millions))
    ax.set_title(
        "Distribution of Price per m² — Market Sweet Spots (orange = IQR)",
        fontsize=14, fontweight="bold",
    )
    ax.set_xlabel("Price per m² (KZT)", fontsize=12)
    ax.set_ylabel("Number of Listings", fontsize=12)
    ax.legend(fontsize=11)
    plt.tight_layout()
    path = OUTPUT_DIR / "5_histogram_price_per_m2.png"
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info("Saved %s", path)


# ═══ Charts 6–10: New-build (новостройка) vs Secondary (вторичка) ═══════════


def chart_nb_bar_mean_price_and_m2(df: pd.DataFrame):
    """6. Grouped comparison: mean total price and mean ₸/m² by segment."""
    d = _with_segment_column(df)
    order = [SEGMENT_NEW, SEGMENT_SEC]
    agg = d.groupby("Segment")[["Price_KZT", "Price_per_m2"]].mean().reindex(order).fillna(0)
    counts = d.groupby("Segment").size().reindex(order).fillna(0).astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors_p = ["#2ecc71", "#e74c3c"]
    colors_m = ["#3498db", "#9b59b6"]

    x = np.arange(len(order))
    axes[0].bar(x, agg["Price_KZT"].values, color=colors_p, edgecolor="white")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{s}\n(n={counts[s]})" for s in order], fontsize=10)
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_millions))
    axes[0].set_title("Mean Price (KZT) by Segment", fontweight="bold")
    axes[0].set_ylabel("KZT")

    axes[1].bar(x, agg["Price_per_m2"].values, color=colors_m, edgecolor="white")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{s}\n(n={counts[s]})" for s in order], fontsize=10)
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_price_per_m2_axis))
    axes[1].set_title("Mean Price per m² by Segment", fontweight="bold")
    axes[1].set_ylabel("₸/m²")

    fig.suptitle(
        "New-Build vs Secondary — Average Price Comparison\n"
        + _newbuild_counts_line(df),
        fontsize=13,
        fontweight="bold",
        y=1.05,
    )
    plt.tight_layout()
    path = OUTPUT_DIR / "6_newbuild_bar_mean_price.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


def chart_nb_boxplot_price_per_m2(df: pd.DataFrame):
    """7. Box plot: ₸/m² distribution by segment."""
    d = _with_segment_column(df)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=d, x="Segment", y="Price_per_m2", order=[SEGMENT_NEW, SEGMENT_SEC], palette=["#2ecc71", "#e74c3c"], showfliers=False, ax=ax)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_price_per_m2_axis))
    ax.set_title(
        "Price per m²: New-Build vs Secondary\n" + _newbuild_counts_line(df),
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("")
    plt.tight_layout()
    path = OUTPUT_DIR / "7_newbuild_boxplot_price_per_m2.png"
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info("Saved %s", path)


def chart_nb_scatter_area_price(df: pd.DataFrame):
    """8. Scatter: area vs price, colored by segment (+ optional Plotly HTML)."""
    d = _with_segment_column(df).dropna(subset=["Area_m2", "Price_KZT"])
    fig = px.scatter(
        d,
        x="Area_m2",
        y="Price_KZT",
        color="Segment",
        color_discrete_map={SEGMENT_NEW: "#2ecc71", SEGMENT_SEC: "#e74c3c"},
        hover_data=["District", "Price_per_m2", "Listing_URL"] if "Listing_URL" in d.columns else ["District", "Price_per_m2"],
        title="Area vs Price — New-Build vs Secondary<br><sup>" + _newbuild_counts_line(df) + "</sup>",
        labels={"Area_m2": "Area (m²)", "Price_KZT": "Price (KZT)"},
        template="plotly_white",
        opacity=0.65,
    )
    fig.write_html(str(OUTPUT_DIR / "8_newbuild_scatter_area_vs_price.html"))
    logger.info("Saved %s", OUTPUT_DIR / "8_newbuild_scatter_area_vs_price.html")

    fig2, ax = plt.subplots(figsize=(11, 7))
    for seg, color in [(SEGMENT_NEW, "#2ecc71"), (SEGMENT_SEC, "#e74c3c")]:
        sub = d[d["Segment"] == seg]
        ax.scatter(sub["Area_m2"], sub["Price_KZT"], label=seg, alpha=0.5, s=28, color=color, edgecolors="none")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_millions))
    ax.set_title(
        "Area vs Price (New-Build vs Secondary)\n" + _newbuild_counts_line(df),
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Area (m²)")
    ax.set_ylabel("Price (KZT)")
    ax.legend(title="Сегмент")
    plt.tight_layout()
    path = OUTPUT_DIR / "8_newbuild_scatter_area_vs_price.png"
    fig2.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig2)
    logger.info("Saved %s", path)


def chart_nb_stacked_counts_by_district(df: pd.DataFrame):
    """9. Stacked bar: listing counts in top districts, split by segment."""
    d = _with_segment_column(df)
    top = d["District"].value_counts().head(8).index.tolist()
    sub = d[d["District"].isin(top)]
    ct = pd.crosstab(sub["District"], sub["Segment"])
    for col in [SEGMENT_NEW, SEGMENT_SEC]:
        if col not in ct.columns:
            ct[col] = 0
    ct = ct[[SEGMENT_NEW, SEGMENT_SEC]]

    fig, ax = plt.subplots(figsize=(12, 6))
    ct.plot(kind="bar", stacked=True, ax=ax, color=["#2ecc71", "#e74c3c"], edgecolor="white", linewidth=0.5)
    ax.set_title(
        "Listings by District — New-Build vs Secondary (stacked)\n" + _newbuild_counts_line(df),
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("District")
    ax.set_ylabel("Number of listings")
    ax.legend(title="Сегмент")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    path = OUTPUT_DIR / "9_newbuild_stacked_counts_by_district.png"
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info("Saved %s", path)


def chart_nb_hist_price_per_m2_overlay(df: pd.DataFrame):
    """10. Overlapping histograms of ₸/m² for both segments."""
    fig, ax = plt.subplots(figsize=(11, 6))
    nb = _with_segment_column(df)
    d_new = nb.loc[nb["Is_NewBuild"], "Price_per_m2"].dropna()
    d_sec = nb.loc[~nb["Is_NewBuild"], "Price_per_m2"].dropna()
    bins = 45
    ax.hist(d_sec, bins=bins, alpha=0.55, color="#e74c3c", label=SEGMENT_SEC, density=True)
    ax.hist(d_new, bins=bins, alpha=0.55, color="#2ecc71", label=SEGMENT_NEW, density=True)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_price_per_m2_axis))
    ax.set_title(
        "Price per m² Distribution (normalized) — New-Build vs Secondary\n" + _newbuild_counts_line(df),
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Price per m² (KZT)")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    path = OUTPUT_DIR / "10_newbuild_hist_price_per_m2_overlay.png"
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info("Saved %s", path)


def visualize(input_path: str = "krisha_cleaned.csv"):
    """
    Generate 5 general market charts + 5 new-build vs secondary charts in charts/.

    Args:
        input_path: Path to cleaned CSV.
    """
    _setup_output_dir()
    df = pd.read_csv(input_path, encoding="utf-8-sig")
    logger.info("Loaded %d rows for visualization", len(df))

    chart_boxplot_price_by_district(df)
    chart_scatter_area_vs_price(df)
    chart_heatmap_correlation(df)
    chart_bar_avg_price_per_m2_by_district(df)
    chart_histogram_price_per_m2(df)

    chart_nb_bar_mean_price_and_m2(df)
    chart_nb_boxplot_price_per_m2(df)
    chart_nb_scatter_area_price(df)
    chart_nb_stacked_counts_by_district(df)
    chart_nb_hist_price_per_m2_overlay(df)

    logger.info("All charts saved to '%s/' (1–10)", OUTPUT_DIR)


if __name__ == "__main__":
    visualize()
