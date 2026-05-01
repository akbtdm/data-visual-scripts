"""
Task 3: Data Analysis — key questions, descriptive stats, correlations, plain-language insights.

At least six structured questions (Q1–Q5 + Q6 “over time”), plus tables suitable for a non-technical audience.
"""

import logging
from typing import Optional, List, Dict, Any

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Human-readable question text (for reports / notebooks)
ANALYTICAL_QUESTIONS: Dict[str, str] = {
    "q1": "В каких районах Алматы самая высокая и самая низкая средняя цена за квадратный метр?",
    "q2": "Связано ли число комнат с ценой за м²? (корреляция и средние по категориям)",
    "q3": "Как позиция этажа (первый, последний, средние) влияет на полную цену квартиры?",
    "q4": "Чем отличается цена в низкой этажности (до 5 этажей) от высотных домов (12+ этажей)?",
    "q5": "Какие микрорайоны / улицы в среднем самые дорогие по ₸/м²?",
    "q6": "Как меняется типичная цена за м² в зависимости от года постройки дома?",
}


def _classify_floor_level(row) -> Optional[str]:
    """Classify floor position as First / Last / Middle."""
    cur = row.get("Current_Floor")
    total = row.get("Total_Floors")
    if pd.isna(cur):
        return None
    if cur == 1:
        return "First"
    if not pd.isna(total) and cur == total:
        return "Last"
    return "Middle"


def _classify_building_type(total_floors) -> Optional[str]:
    """Khrushchevka (4-5 floors) vs Modern High-rise (12+ floors)."""
    if pd.isna(total_floors):
        return None
    if total_floors <= 5:
        return "Khrushchevka (≤5 floors)"
    if total_floors >= 12:
        return "Modern High-rise (12+ floors)"
    return "Mid-rise (6-11 floors)"


def _coerce_bool_series(s: pd.Series) -> pd.Series:
    def to_bool(x) -> bool:
        if pd.isna(x):
            return False
        if isinstance(x, (bool, np.bool_)):
            return bool(x)
        t = str(x).strip().lower()
        return t in ("true", "1", "yes", "1.0")

    return s.map(to_bool)


def analyze(input_path: str = "krisha_cleaned.csv") -> dict:
    """
    Run analytical questions, descriptive statistics, and correlations.

    Args:
        input_path: Path to cleaned CSV.

    Returns:
        Dict with keys:
          - ``questions`` — short Russian formulations (ANALYTICAL_QUESTIONS)
          - ``describe_numeric`` — ``DataFrame.describe()`` for core numerics
          - ``correlations`` — Pearson matrix for key numeric columns
          - ``q1`` … ``q6`` — answer tables / structures
    """
    df = pd.read_csv(input_path, encoding="utf-8-sig")
    logger.info("Loaded %d rows for analysis", len(df))

    results: Dict[str, Any] = {"questions": dict(ANALYTICAL_QUESTIONS)}

    if "Is_NewBuild" in df.columns:
        df = df.copy()
        df["Is_NewBuild"] = _coerce_bool_series(df["Is_NewBuild"])

    # ── Descriptive statistics (core numerics) ─────────────────────────────
    num_core = [
        c
        for c in (
            "Price_KZT",
            "Price_per_m2",
            "Area_m2",
            "Rooms",
            "Current_Floor",
            "Total_Floors",
            "Build_Year",
            "Ceiling_Height",
        )
        if c in df.columns
    ]
    results["describe_numeric"] = df[num_core].describe(percentiles=[0.25, 0.5, 0.75]).round(2)
    logger.info("Descriptive statistics computed for %d numeric columns", len(num_core))

    # ── Correlation matrix (factors influencing price / m²) ─────────────────
    corr_cols = [c for c in ("Price_per_m2", "Price_KZT", "Area_m2", "Rooms", "Current_Floor", "Total_Floors", "Build_Year", "Ceiling_Height") if c in df.columns]
    corr_df = df[corr_cols].dropna(how="all").corr(numeric_only=True).round(3)
    results["correlations"] = corr_df
    logger.info("Correlation matrix (%d×%d)", len(corr_cols), len(corr_cols))

    # ── Q1: Average Price per m2 by District ────────────────────────────────
    q1 = (
        df.groupby("District")["Price_per_m2"]
        .agg(["mean", "median", "count"])
        .rename(columns={"mean": "Avg_Price_per_m2", "median": "Median_Price_per_m2", "count": "Listings"})
        .sort_values("Avg_Price_per_m2", ascending=False)
        .round(0)
    )
    results["q1"] = q1
    logger.info("Q1 — Districts ranked by avg price/m2:\n%s", q1.to_string())

    # ── Q2: Correlation between Rooms and Price_per_m2 ──────────────────────
    valid = df[["Rooms", "Price_per_m2"]].dropna()
    corr_pearson = valid["Rooms"].corr(valid["Price_per_m2"])
    corr_spearman = valid["Rooms"].corr(valid["Price_per_m2"], method="spearman")
    room_groups = (
        valid.groupby("Rooms")["Price_per_m2"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "Avg_Price_per_m2", "count": "Listings"})
        .round(0)
    )
    results["q2"] = {
        "pearson_corr": round(corr_pearson, 4),
        "spearman_corr": round(corr_spearman, 4),
        "by_room_count": room_groups,
    }
    logger.info(
        "Q2 — Rooms vs Price/m2: Pearson=%.4f, Spearman=%.4f",
        corr_pearson, corr_spearman,
    )

    # ── Q3: Most expensive floor level (First / Middle / Last) ──────────────
    df["Floor_Level"] = df.apply(_classify_floor_level, axis=1)
    q3 = (
        df.dropna(subset=["Floor_Level"])
        .groupby("Floor_Level")["Price_KZT"]
        .agg(["mean", "median", "count"])
        .rename(columns={"mean": "Avg_Price_KZT", "median": "Median_Price_KZT", "count": "Listings"})
        .sort_values("Avg_Price_KZT", ascending=False)
        .round(0)
    )
    results["q3"] = q3
    logger.info("Q3 — Floor level prices:\n%s", q3.to_string())

    # ── Q4: Khrushchevka vs Modern High-rise prices ──────────────────────────
    df["Building_Type"] = df["Total_Floors"].apply(_classify_building_type)
    q4 = (
        df.dropna(subset=["Building_Type"])
        .groupby("Building_Type")[["Price_KZT", "Price_per_m2"]]
        .agg(["mean", "median", "count"])
        .round(0)
    )
    q4.columns = ["_".join(c) for c in q4.columns]
    results["q4"] = q4
    logger.info("Q4 — Building types:\n%s", q4.to_string())

    # ── Q5: Top 5 most expensive streets/micro-districts ───────────────────
    # Extract micro-district / street from Address
    df["Microdistrict"] = df["Address"].str.split(",").str[0].str.strip()
    q5 = (
        df.groupby("Microdistrict")["Price_per_m2"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "Avg_Price_per_m2", "count": "Listings"})
        .query("Listings >= 3")          # min 3 listings to be meaningful
        .sort_values("Avg_Price_per_m2", ascending=False)
        .head(10)
        .round(0)
    )
    results["q5"] = q5
    logger.info("Q5 — Top 10 micro-districts by price/m2:\n%s", q5.to_string())

    # ── Q6: Price per m² vs year of construction (“change over time” / cohorts) ─
    ydf = df.dropna(subset=["Build_Year"]).copy()
    if not ydf.empty:
        q6 = (
            ydf.groupby("Build_Year")["Price_per_m2"]
            .agg(["median", "mean", "count"])
            .rename(columns={"median": "Median_Price_per_m2", "mean": "Avg_Price_per_m2", "count": "Listings"})
            .query("Listings >= 3")
            .sort_index()
            .round(0)
        )
    else:
        q6 = pd.DataFrame()
    results["q6"] = q6
    if not q6.empty:
        logger.info("Q6 — Price/m2 by build year (min 3 listings/year):\n%s", q6.to_string())

    results["plain_language"] = plain_language_insights(results)

    return results


def plain_language_insights(results: dict) -> List[str]:
    """
    Short bullet-style takeaways for non-technical readers (Russian).
    Expects a ``results`` dict as returned by :func:`analyze` (without nesting ``plain_language``).
    """
    lines: List[str] = []

    q1: pd.DataFrame = results.get("q1", pd.DataFrame())
    if isinstance(q1, pd.DataFrame) and len(q1) >= 2:
        q1f = q1.drop(index="Прочее", errors="ignore")
        if len(q1f) < 2:
            q1f = q1
        top, bottom = q1f.index[0], q1f.index[-1]
        dv = (q1f.loc[top, "Avg_Price_per_m2"] - q1f.loc[bottom, "Avg_Price_per_m2"]) / q1f.loc[
            bottom, "Avg_Price_per_m2"
        ] * 100
        lines.append(
            f"По средней цене за м² самый дорогой район — «{top}», самый доступный среди основных — «{bottom}» "
            f"(разница порядка {dv:.0f}%)."
        )

    q2 = results.get("q2") or {}
    if isinstance(q2, dict) and "pearson_corr" in q2:
        r = q2["pearson_corr"]
        if r > 0.15:
            lines.append(
                "Больше комнат в среднем связано с более высокой ценой за м² — типично для крупных и премиальных квартир."
            )
        elif r < -0.15:
            lines.append(
                "Больше комнат в среднем связано с более низкой ценой за м² — малые квартиры часто «дороже за метр»."
            )
        else:
            lines.append(
                "Связь числа комнат с ценой за м² умеренная: на цену сильнее влияют район, этажность дома и локация."
            )

    q3: pd.DataFrame = results.get("q3", pd.DataFrame())
    if isinstance(q3, pd.DataFrame) and not q3.empty:
        best = q3.index[0]
        lines.append(
            f"Среди объявлений с известным этажом самая высокая средняя полная цена — у категории «{best}» этажа."
        )

    q4: pd.DataFrame = results.get("q4", pd.DataFrame())
    if isinstance(q4, pd.DataFrame) and not q4.empty:
        lines.append(
            "Низкоэтажные дома (до 5 этажей) и высотки (12+) заметно отличаются по средней цене за м² — см. таблицу Q4."
        )

    q5: pd.DataFrame = results.get("q5", pd.DataFrame())
    if isinstance(q5, pd.DataFrame) and not q5.empty:
        lines.append(
            f"Среди микрорайонов с несколькими объявлениями лидер по ₸/м² — «{q5.index[0]}»."
        )

    q6: pd.DataFrame = results.get("q6", pd.DataFrame())
    if isinstance(q6, pd.DataFrame) and len(q6) >= 2:
        lines.append(
            "По годам постройки видно, как меняется типичная (медианная) цена за м² — это помогает сравнивать «молодые» дома и более старые cohorts."
        )
    elif isinstance(q6, pd.DataFrame) and len(q6) == 1:
        lines.append(
            "В данных мало разных годов постройки с достаточным числом объявлений; для выводов «во времени» лучше расширить выборку."
        )

    return lines


if __name__ == "__main__":
    res = analyze()
    print("\n=== Questions (formulations) ===")
    for k, v in res["questions"].items():
        print(f"  {k}: {v}")
    print("\n=== Descriptive statistics (numeric) ===")
    print(res["describe_numeric"])
    print("\n=== Correlations ===")
    print(res["correlations"])
    print("\n=== Q1: Avg Price/m2 by District ===")
    print(res["q1"])
    print("\n=== Q2: Rooms vs Price/m2 Correlation ===")
    print(f"Pearson: {res['q2']['pearson_corr']}, Spearman: {res['q2']['spearman_corr']}")
    print(res["q2"]["by_room_count"])
    print("\n=== Q3: Floor Level Prices ===")
    print(res["q3"])
    print("\n=== Q4: Building Type Prices ===")
    print(res["q4"])
    print("\n=== Q5: Top 10 Most Expensive Micro-districts ===")
    print(res["q5"])
    print("\n=== Q6: Price/m2 by Build Year (min 3 listings) ===")
    print(res["q6"])
    print("\n=== Plain-language insights ===")
    for line in res["plain_language"]:
        print(" •", line)
