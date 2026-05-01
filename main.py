"""
INF 369 Final Project — Data-Driven Analysis of the Almaty Housing Market (Krisha.kz)
Full catalog scrape with Is_NewBuild labels; 10 charts (5 general + 5 new vs secondary).
Orchestrates: Scrape → Clean → Analyze → Visualize → Report
"""

import sys
import logging
from pathlib import Path

import pandas as pd

from scraper import scrape
from preprocessing import clean
from analysis import analyze, ANALYTICAL_QUESTIONS
from visualization import visualize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

RAW_CSV = "krisha_raw.csv"
CLEAN_CSV = "krisha_cleaned.csv"
# None = crawl every page until Krisha returns empty (within SCRAPE_MAX_PAGES_CAP per pass)
SCRAPE_PAGES = None
SCRAPE_MAX_PAGES_CAP = 10_000
BUILD_YEAR_SPAN = 2  # Krisha das[house.year]: last N calendar years through current year
SCRAPE_MAX_LISTINGS = 2000  # Cap on total Pass B records (None = no cap, full crawl).


# ── Task 5: Storytelling Summary Report ─────────────────────────────────────
def print_summary_report(results: dict, df: pd.DataFrame):
    """Print key insights and 'turning points' for presentation slides."""

    sep = "=" * 70
    print(f"\n{sep}")
    print("  ALMATY HOUSING MARKET — KEY INSIGHTS & TURNING POINTS")
    print(sep)

    # ── Plain-language bullets (non-technical audience) ────────────────────
    for line in results.get("plain_language", []):
        print(f"\n  • {line}")

    # ── Insight 1: Cheapest vs most expensive district ─────────────────────
    q1: pd.DataFrame = results["q1"]
    if len(q1) >= 2:
        most_exp = q1.index[0]
        cheapest = q1.index[-1]
        most_exp_val = q1.loc[most_exp, "Avg_Price_per_m2"]
        cheapest_val = q1.loc[cheapest, "Avg_Price_per_m2"]
        pct_diff = (most_exp_val - cheapest_val) / cheapest_val * 100
        print(f"\n[District Prices]")
        print(f"  • Most expensive district : {most_exp} — {most_exp_val:,.0f} ₸/m²")
        print(f"  • Most affordable district: {cheapest} — {cheapest_val:,.0f} ₸/m²")
        print(f"  ★ TURNING POINT: {most_exp} is {pct_diff:.0f}% MORE expensive than {cheapest}")

        # Check if a "central-sounding" district is unexpectedly cheap
        central_keywords = ["Алмалинский", "Медеуский", "Бостандыкский"]
        for kw in central_keywords:
            match = [d for d in q1.index if kw in d]
            if match:
                d = match[0]
                rank = list(q1.index).index(d) + 1
                print(f"  ★ {d} ranks #{rank} out of {len(q1)} districts by price/m²")

    # ── Insight 2: Rooms vs Price correlation ──────────────────────────────
    q2 = results["q2"]
    pearson = q2["pearson_corr"]
    direction = "NEGATIVE" if pearson < 0 else "POSITIVE"
    strength = "weak" if abs(pearson) < 0.3 else ("moderate" if abs(pearson) < 0.6 else "strong")
    print(f"\n[Rooms vs Price/m²]")
    print(f"  • Pearson correlation: {pearson:.3f} ({strength} {direction})")
    if pearson < -0.1:
        print("  ★ TURNING POINT: More rooms = CHEAPER per m² — smaller units command a premium!")
    elif pearson > 0.1:
        print("  ★ TURNING POINT: Larger apartments command higher price per m² — luxury effect.")
    else:
        print("  ★ Room count has little independent effect on price/m².")

    # ── Insight 3: Floor level ─────────────────────────────────────────────
    q3: pd.DataFrame = results["q3"]
    if not q3.empty:
        top_floor_level = q3.index[0]
        print(f"\n[Floor Level Analysis]")
        for level in q3.index:
            print(f"  • {level} floor: avg {q3.loc[level,'Avg_Price_KZT']:,.0f} ₸")
        print(f"  ★ TURNING POINT: {top_floor_level} floors are the most expensive on average.")

    # ── Insight 4: Khrushchevka vs Modern ─────────────────────────────────
    q4: pd.DataFrame = results["q4"]
    if not q4.empty:
        print(f"\n[Building Type Comparison]")
        types = q4.index.tolist()
        khrush = [t for t in types if "Khrushchevka" in t]
        modern = [t for t in types if "High-rise" in t]
        if khrush and modern:
            k_price = q4.loc[khrush[0], "Price_per_m2_mean"]
            m_price = q4.loc[modern[0], "Price_per_m2_mean"]
            pct = (m_price - k_price) / k_price * 100
            print(f"  • Khrushchevka (≤5 fl.) : avg {k_price:,.0f} ₸/m²")
            print(f"  • Modern High-rise (12+) : avg {m_price:,.0f} ₸/m²")
            if pct > 0:
                print(f"  ★ TURNING POINT: Modern high-rises cost {pct:.0f}% MORE than Khrushchevkas per m².")
            else:
                print(f"  ★ TURNING POINT: Khrushchevkas actually cost {abs(pct):.0f}% MORE per m² — "
                      f"likely due to central location premium!")

    # ── Insight 5: Top streets ─────────────────────────────────────────────
    q5: pd.DataFrame = results["q5"]
    if not q5.empty:
        print(f"\n[Top 5 Most Expensive Micro-districts/Streets]")
        for i, (idx, row) in enumerate(q5.head(5).iterrows(), 1):
            print(f"  {i}. {idx} — {row['Avg_Price_per_m2']:,.0f} ₸/m²  ({int(row['Listings'])} listings)")
        print(f"  ★ The #1 most expensive location is '{q5.index[0]}'")

    q6: pd.DataFrame = results.get("q6", pd.DataFrame())
    if isinstance(q6, pd.DataFrame) and not q6.empty:
        print(f"\n[Build year vs price/m² — Q6: {ANALYTICAL_QUESTIONS.get('q6', '')}]")
        print(q6.to_string())
        if len(q6) >= 2:
            y_lo, y_hi = q6["Median_Price_per_m2"].idxmin(), q6["Median_Price_per_m2"].idxmax()
            print(f"  ★ Lowest median ₸/m² cohort: build year {y_lo}; highest: {y_hi} (among years with ≥3 listings).")

    # ── General market stats ───────────────────────────────────────────────
    print(f"\n[Market Overview]")
    print(f"  • Total listings analyzed : {len(df):,}")
    print(f"  • Median price            : {df['Price_KZT'].median():,.0f} ₸")
    print(f"  • Median price per m²     : {df['Price_per_m2'].median():,.0f} ₸/m²")
    print(f"  • Median area             : {df['Area_m2'].median():.1f} m²")
    print(f"  • Districts covered       : {df['District'].nunique()}")

    if "Is_NewBuild" in df.columns:
        nb = df["Is_NewBuild"].astype(bool)
        n_nb, n_sec = int(nb.sum()), int((~nb).sum())
        print(f"\n[New build vs secondary (labeled)]")
        print(f"  • Новостройка : {n_nb:,} listings ({100 * n_nb / len(df):.1f}%)")
        print(f"  • Вторичка    : {n_sec:,} listings ({100 * n_sec / len(df):.1f}%)")
        if n_nb > 0 and n_sec > 0:
            m_nb = df.loc[nb, "Price_per_m2"].median()
            m_sec = df.loc[~nb, "Price_per_m2"].median()
            if m_sec > 0:
                pct = (m_nb - m_sec) / m_sec * 100
                print(f"  • Median ₸/m²: новостройка {m_nb:,.0f} vs вторичка {m_sec:,.0f} ({pct:+.1f}% vs secondary)")
            print("  ★ TURNING POINT: Compare segments in charts 6–10 (new-build split).")

    print(f"\n{sep}")
    print("  Charts: run `visualization.py` for the 10 PNG/HTML set, or `analysis.ipynb` (Run All) for the extended chart pack in charts/.")
    print("  Full pipeline log in 'pipeline.log'.")
    print(sep)


def main():
    logger.info("╔══ Starting Almaty Housing Market Analysis Pipeline ══╗")

    # ── Task 1: Scrape ───────────────────────────────────────────────────────
    if Path(RAW_CSV).exists():
        logger.info("'%s' already exists — skipping scrape. Delete to re-scrape.", RAW_CSV)
    else:
        if SCRAPE_PAGES is None:
            logger.info(
                "Task 1: Full crawl — year window %d yrs, 2 passes, until empty (cap %s pages/pass)...",
                BUILD_YEAR_SPAN,
                f"{SCRAPE_MAX_PAGES_CAP:,}",
            )
        else:
            logger.info(
                "Task 1: Scraping krisha.kz (%d pages × 2 passes, year window %d yrs)...",
                SCRAPE_PAGES,
                BUILD_YEAR_SPAN,
            )
        scrape(
            pages=SCRAPE_PAGES,
            output_path=RAW_CSV,
            year_span=BUILD_YEAR_SPAN,
            max_pages_cap=SCRAPE_MAX_PAGES_CAP,
            max_listings=SCRAPE_MAX_LISTINGS,
        )

    raw_df = pd.read_csv(RAW_CSV, encoding="utf-8-sig")
    logger.info("Raw data: %d rows", len(raw_df))
    if len(raw_df) < 500:
        logger.warning("Only %d rows in raw data — for a course sample you may want more pages or a wider year window.", len(raw_df))

    # ── Task 2: Clean ────────────────────────────────────────────────────────
    logger.info("Task 2: Cleaning data...")
    df = clean(input_path=RAW_CSV, output_path=CLEAN_CSV)

    # ── Task 3: Analyze ──────────────────────────────────────────────────────
    logger.info("Task 3: Running analysis...")
    results = analyze(input_path=CLEAN_CSV)

    # ── Task 4: Visualize ────────────────────────────────────────────────────
    logger.info("Task 4: Generating visualizations...")
    visualize(input_path=CLEAN_CSV)

    # ── Task 5: Summary Report ────────────────────────────────────────────────
    print_summary_report(results, df)

    logger.info("╚══ Pipeline complete ══╝")


if __name__ == "__main__":
    main()
