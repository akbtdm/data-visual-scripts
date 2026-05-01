"""
Task 2: Data Preprocessing — Clean krisha_raw.csv into krisha_cleaned.csv

Rubric alignment (INF 369-style “clean the data”):

**Null values**
  - *Remove rows* where core fields are missing: ``Price_KZT``, ``Area_m2`` (cannot compute ₸/m²).
  - *Impute / replace*: ``Rooms`` → median; ``Address`` → «Неизвестный»; ``Description_Snippet`` → "";
    optional text fields stay NA until grouped (see ``House_Type_Group``, ``Bathroom_Group``).
  - Floor fields stay nullable (unknown floor is informative absence, not filled with a fake number).

**Duplicates & outliers**
  - Drop exact duplicates on ``Listing_URL``, then on (``Price_KZT``, ``Area_m2``, ``Address``).
  - Hard rules: ``Area_m2`` < 10; ``Price_KZT`` > 1e9 or ≤ 0.
  - Soft rule: keep rows between the 1st and 99th *percentile* of ``Price_KZT`` (trim extreme tails).

**Columns for analysis (grouping / encoding)**
  - ``Market_Segment`` — human-readable new vs secondary.
  - ``Price_Tier`` — tertiles of ``Price_per_m2``.
  - ``House_Type_Group``, ``Bathroom_Group`` — rare categories collapsed to «Прочее».
  - ``Build_Year_Bucket`` — era bands for time-style questions.
  - ``Area_Band`` — quartiles of area.
  - ``District_Code``, ``Is_NewBuild_Code`` — integer codes for models / heatmaps.

Also: parse ``Floor_Info``, normalize ``District``, derive ``Price_per_m2``.
"""

import re
import logging
from typing import Optional, Tuple

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _coerce_bool_newbuild(series: pd.Series) -> pd.Series:
    """CSV-safe: strings 'False' must not become True (pandas astype(bool) is wrong for strings)."""

    def to_bool(x) -> bool:
        if pd.isna(x):
            return False
        if isinstance(x, (bool, np.bool_)):
            return bool(x)
        s = str(x).strip().lower()
        return s in ("true", "1", "yes", "1.0")

    return series.map(to_bool)


def _parse_floor_info(floor_str) -> Tuple[Optional[int], Optional[int]]:
    """
    Parse various Floor_Info formats:
      '5 из 9'  → (5, 9)
      '5/9 этаж' → (5, 9)
      '5'       → (5, None)
      NaN       → (None, None)
    """
    if pd.isna(floor_str) or not str(floor_str).strip():
        return None, None
    s = str(floor_str).strip()
    # Format: 'N из M'
    m = re.match(r"(\d+)\s+из\s+(\d+)", s)
    if m:
        return int(m.group(1)), int(m.group(2))
    # Format: 'N/M'
    m = re.match(r"(\d+)/(\d+)", s)
    if m:
        return int(m.group(1)), int(m.group(2))
    # Single floor number
    m = re.match(r"(\d+)", s)
    if m:
        return int(m.group(1)), None
    return None, None


def clean(input_path: str = "krisha_raw.csv", output_path: str = "krisha_cleaned.csv") -> pd.DataFrame:
    """
    Clean raw scraped data and save to CSV.

    Args:
        input_path:  Path to raw CSV.
        output_path: Path for cleaned CSV.

    Returns:
        Cleaned DataFrame.
    """
    df = pd.read_csv(input_path, encoding="utf-8-sig")
    initial_count = len(df)
    logger.info("Loaded %d rows from '%s'", initial_count, input_path)

    # ── Drop legacy filter-window columns: not per-listing data ──────────────
    legacy_filter_cols = [c for c in ("Filter_Year_From", "Filter_Year_To") if c in df.columns]
    if legacy_filter_cols:
        df = df.drop(columns=legacy_filter_cols)
        logger.info("Dropped legacy filter columns: %s", ", ".join(legacy_filter_cols))

    # ── 0. New-build flag (scraper) + legacy CSV fallback ────────────────────
    if "Is_NewBuild" not in df.columns:
        df["Is_NewBuild"] = False
    else:
        df["Is_NewBuild"] = _coerce_bool_newbuild(df["Is_NewBuild"])

    # Text heuristic only for legacy exports without Listing_ID (no official Krisha split).
    if len(df) > 0 and df["Is_NewBuild"].sum() == 0 and "Listing_ID" not in df.columns:
        from scraper import infer_newbuild_from_text

        logger.warning(
            "Legacy CSV: no Listing_ID / all Is_NewBuild False — inferring новостройка from text. "
            "Re-scrape for labels from Krisha «новостройки» (official IDs)."
        )
        df["Is_NewBuild"] = df.apply(
            lambda r: infer_newbuild_from_text(
                str(r.get("Description_Snippet", "") or ""),
                str(r.get("Title", "") or ""),
            ),
            axis=1,
        )
        logger.info(
            "Heuristic labels: %d новостройка, %d вторичка",
            int(df["Is_NewBuild"].sum()),
            int(len(df) - int(df["Is_NewBuild"].sum())),
        )
    elif len(df) > 0 and df["Is_NewBuild"].sum() == 0 and "Listing_ID" in df.columns:
        logger.info(
            "Is_NewBuild all False — keeping scraper labels (вся выборка не в novostroiki ID set, т.е. вторичка в разрезе Крыши)."
        )

    # ── 1. Parse Floor_Info ─────────────────────────────────────────────────
    floors = df["Floor_Info"].apply(_parse_floor_info)
    df["Current_Floor"] = floors.apply(lambda x: x[0]).astype("Int64")
    df["Total_Floors"] = floors.apply(lambda x: x[1]).astype("Int64")

    # ── 2. Ensure numeric types ──────────────────────────────────────────────
    df["Price_KZT"] = pd.to_numeric(df["Price_KZT"], errors="coerce")
    df["Area_m2"] = pd.to_numeric(df["Area_m2"], errors="coerce")
    df["Rooms"] = pd.to_numeric(df["Rooms"], errors="coerce").astype("Int64")

    # ── 2b. Type-coerce new per-listing facts ────────────────────────────────
    if "Build_Year" in df.columns:
        years = pd.to_numeric(df["Build_Year"], errors="coerce")
        current_year = pd.Timestamp.today().year
        years = years.where((years >= 1900) & (years <= current_year + 5))
        df["Build_Year"] = years.astype("Int64")
    if "Ceiling_Height" in df.columns:
        heights = pd.to_numeric(df["Ceiling_Height"], errors="coerce")
        df["Ceiling_Height"] = heights.where((heights >= 1.5) & (heights <= 6.0))
    for text_col in ("Residential_Complex", "House_Type", "Bathroom", "Developer"):
        if text_col in df.columns:
            df[text_col] = (
                df[text_col]
                .astype("string")
                .str.strip()
                .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
            )

    # ── 3. Drop rows missing critical numeric fields ─────────────────────────
    df.dropna(subset=["Price_KZT", "Area_m2"], inplace=True)
    logger.info("After dropping missing Price/Area: %d rows", len(df))

    # ── 4. Remove exact duplicates (same URL or same price+area+address) ────
    df.drop_duplicates(subset=["Listing_URL"], keep="first", inplace=True)
    df.drop_duplicates(subset=["Price_KZT", "Area_m2", "Address"], keep="first", inplace=True)
    logger.info("After deduplication: %d rows", len(df))

    # ── 5. Remove hard outliers ──────────────────────────────────────────────
    # Micro-units under 10 m² are likely data errors
    df = df[df["Area_m2"] >= 10]
    # Prices over 1 billion KZT are ultra-luxury or errors — remove for analysis
    df = df[df["Price_KZT"] <= 1_000_000_000]
    # Sanity check: price must be positive
    df = df[df["Price_KZT"] > 0]
    logger.info("After hard outlier removal: %d rows", len(df))

    # ── 6. IQR-based outlier removal on Price_KZT ───────────────────────────
    Q1 = df["Price_KZT"].quantile(0.01)
    Q3 = df["Price_KZT"].quantile(0.99)
    df = df[(df["Price_KZT"] >= Q1) & (df["Price_KZT"] <= Q3)]
    logger.info("After IQR outlier removal (1-99 pct) on price: %d rows", len(df))

    # ── 7. Derive Price_per_m2 ───────────────────────────────────────────────
    df["Price_per_m2"] = (df["Price_KZT"] / df["Area_m2"]).round(0)

    # ── 8. Normalize District to canonical Almaty district names ─────────────
    KNOWN_DISTRICTS = {
        "Алатауский": "Алатауский р-н",
        "Алмалинский": "Алмалинский р-н",
        "Ауэзовский": "Ауэзовский р-н",
        "Бостандыкский": "Бостандыкский р-н",
        "Жетысуский": "Жетысуский р-н",
        "Медеуский": "Медеуский р-н",
        "Наурызбайский": "Наурызбайский р-н",
        "Турксибский": "Турксибский р-н",
    }

    def _normalize_district(d):
        if pd.isna(d):
            return "Прочее"
        for key, canonical in KNOWN_DISTRICTS.items():
            if key in str(d):
                return canonical
        return "Прочее"

    df["District"] = df["District"].apply(_normalize_district)

    # ── 9. Fill remaining nulls ──────────────────────────────────────────────
    df["Rooms"] = df["Rooms"].fillna(df["Rooms"].median())
    df["Address"] = df["Address"].fillna("Неизвестный")
    df["Description_Snippet"] = df["Description_Snippet"].fillna("")

    # Floor columns: leave as nullable Int64 (NaN acceptable for analysis)

    # ── 9b. Analysis-ready categoricals (grouping + light encoding) ───────────
    df["Market_Segment"] = np.where(df["Is_NewBuild"], "Новостройка", "Вторичка")

    try:
        df["Price_Tier"] = pd.qcut(
            df["Price_per_m2"],
            q=3,
            labels=["Недорогой сегмент", "Средний сегмент", "Премиум сегмент"],
            duplicates="drop",
        )
    except (ValueError, TypeError):
        df["Price_Tier"] = pd.Series(pd.NA, index=df.index, dtype="object")

    MIN_HOUSE_N = max(8, int(len(df) * 0.01))
    if "House_Type" in df.columns:
        ht = df["House_Type"].fillna("Не указано").astype(str).str.strip()
        vc_ht = ht.value_counts()
        rare_ht = set(vc_ht[vc_ht < MIN_HOUSE_N].index)
        df["House_Type_Group"] = ht.where(~ht.isin(rare_ht), "Прочее / редкий тип")
    else:
        df["House_Type_Group"] = "Не указано"

    MIN_BATH_N = max(10, int(len(df) * 0.015))
    if "Bathroom" in df.columns:
        bg = df["Bathroom"].fillna("Не указано").astype(str).str.strip()
        vc_b = bg.value_counts()
        rare_b = set(vc_b[vc_b < MIN_BATH_N].index) - {"Не указано"}
        df["Bathroom_Group"] = bg.where(~bg.isin(rare_b), "Прочее")
    else:
        df["Bathroom_Group"] = "Не указано"

    def _year_bucket(y) -> str:
        if pd.isna(y):
            return "Не указан"
        yi = int(y)
        if yi < 2000:
            return "до 2000"
        if yi < 2010:
            return "2000–2009"
        if yi < 2016:
            return "2010–2015"
        if yi < 2020:
            return "2016–2019"
        if yi < 2024:
            return "2020–2023"
        return "2024 и новее"

    if "Build_Year" in df.columns:
        df["Build_Year_Bucket"] = df["Build_Year"].apply(_year_bucket)
    else:
        df["Build_Year_Bucket"] = "Не указан"

    try:
        df["Area_Band"] = pd.qcut(
            df["Area_m2"],
            q=4,
            labels=["Маленькая", "Ниже средней", "Выше средней", "Большая"],
            duplicates="drop",
        )
    except (ValueError, TypeError):
        df["Area_Band"] = pd.Series(pd.NA, index=df.index, dtype="object")

    df["District_Code"] = pd.Categorical(df["District"]).codes
    df["Is_NewBuild_Code"] = df["Is_NewBuild"].astype(int)

    # ── 10. Reset index & save ───────────────────────────────────────────────
    df.reset_index(drop=True, inplace=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    removed = initial_count - len(df)
    logger.info(
        "Cleaned dataset: %d rows (removed %d / %.1f%%)",
        len(df), removed, removed / initial_count * 100
    )
    logger.info("Saved to '%s'", output_path)
    return df


if __name__ == "__main__":
    df = clean()
    print(df.describe())
    print("\nNull counts:\n", df.isnull().sum())
