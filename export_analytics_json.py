"""
Export the analysis results and the cleaned dataset into JSON for the React front-end.

Outputs (in front/public/data/):
- summary.json           — high-level KPIs + descriptive statistics
- correlations.json      — correlation matrix
- q1_districts.json      — districts ranked by avg price/m²
- q2_rooms.json          — rooms vs price/m² + correlations
- q3_floor_level.json    — first / middle / last floor pricing
- q4_building_type.json  — khrushchevka vs high-rise pricing
- q5_microdistricts.json — top expensive micro-districts
- q6_build_year.json     — price/m² by build year
- insights.json          — plain-language takeaways
- listings.json          — cleaned dataset (compact, for the explorer table & scatter)
- questions.json         — the analytical questions (RU)
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis import analyze, ANALYTICAL_QUESTIONS

OUTPUT_DIR = Path("front/public/data")
CSV_PATH = "krisha_cleaned.csv"


def _clean(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, (np.floating,)):
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _df_records(df: pd.DataFrame, index_name: str | None = None) -> list[dict]:
    out: list[dict] = []
    for idx, row in df.iterrows():
        rec = {k: _clean(v) for k, v in row.items()}
        if index_name is not None:
            rec[index_name] = _clean(idx)
        out.append(rec)
    return out


def _df_to_matrix(df: pd.DataFrame) -> dict:
    return {
        "columns": list(df.columns),
        "index": [str(i) for i in df.index],
        "values": [[_clean(v) for v in row] for row in df.values.tolist()],
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    res = analyze(CSV_PATH)
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")

    # Coerce booleans for consistency
    if "Is_NewBuild" in df.columns:
        df["Is_NewBuild"] = df["Is_NewBuild"].astype(str).str.lower().isin(["true", "1", "1.0", "yes"])

    # ── summary.json ────────────────────────────────────────────────────────
    n = len(df)
    new_n = int(df["Is_NewBuild"].sum()) if "Is_NewBuild" in df.columns else 0
    sec_n = n - new_n
    districts = sorted(df["District"].dropna().unique().tolist())
    summary = {
        "totalListings": n,
        "newBuilds": new_n,
        "secondary": sec_n,
        "districts": districts,
        "districtCount": len(districts),
        "medianPrice": _clean(df["Price_KZT"].median()),
        "meanPrice": _clean(df["Price_KZT"].mean()),
        "medianPricePerM2": _clean(df["Price_per_m2"].median()),
        "meanPricePerM2": _clean(df["Price_per_m2"].mean()),
        "medianArea": _clean(df["Area_m2"].median()),
        "meanArea": _clean(df["Area_m2"].mean()),
        "minPrice": _clean(df["Price_KZT"].min()),
        "maxPrice": _clean(df["Price_KZT"].max()),
        "minPricePerM2": _clean(df["Price_per_m2"].min()),
        "maxPricePerM2": _clean(df["Price_per_m2"].max()),
        "describe": _df_to_matrix(res["describe_numeric"]),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")

    # ── correlations ────────────────────────────────────────────────────────
    (OUTPUT_DIR / "correlations.json").write_text(
        json.dumps(_df_to_matrix(res["correlations"]), ensure_ascii=False),
        encoding="utf-8",
    )

    # ── q1: districts ───────────────────────────────────────────────────────
    q1 = res["q1"].reset_index().rename(columns={"District": "district"})
    (OUTPUT_DIR / "q1_districts.json").write_text(
        json.dumps([
            {
                "district": str(r["district"]),
                "avgPricePerM2": _clean(r["Avg_Price_per_m2"]),
                "medianPricePerM2": _clean(r["Median_Price_per_m2"]),
                "listings": _clean(r["Listings"]),
            }
            for _, r in q1.iterrows()
        ], ensure_ascii=False),
        encoding="utf-8",
    )

    # ── q2: rooms ───────────────────────────────────────────────────────────
    rooms_df = res["q2"]["by_room_count"].reset_index()
    (OUTPUT_DIR / "q2_rooms.json").write_text(
        json.dumps({
            "pearson": _clean(res["q2"]["pearson_corr"]),
            "spearman": _clean(res["q2"]["spearman_corr"]),
            "byRoomCount": [
                {
                    "rooms": _clean(r["Rooms"]),
                    "avgPricePerM2": _clean(r["Avg_Price_per_m2"]),
                    "listings": _clean(r["Listings"]),
                }
                for _, r in rooms_df.iterrows()
            ],
        }, ensure_ascii=False),
        encoding="utf-8",
    )

    # ── q3: floor level ─────────────────────────────────────────────────────
    q3 = res["q3"].reset_index().rename(columns={"Floor_Level": "floorLevel"})
    (OUTPUT_DIR / "q3_floor_level.json").write_text(
        json.dumps([
            {
                "floorLevel": str(r["floorLevel"]),
                "avgPrice": _clean(r["Avg_Price_KZT"]),
                "medianPrice": _clean(r["Median_Price_KZT"]),
                "listings": _clean(r["Listings"]),
            }
            for _, r in q3.iterrows()
        ], ensure_ascii=False),
        encoding="utf-8",
    )

    # ── q4: building type ───────────────────────────────────────────────────
    q4 = res["q4"].reset_index().rename(columns={"Building_Type": "buildingType"})
    q4_records = []
    for _, r in q4.iterrows():
        rec = {"buildingType": str(r["buildingType"])}
        for col in q4.columns:
            if col == "buildingType":
                continue
            rec[col] = _clean(r[col])
        q4_records.append(rec)
    (OUTPUT_DIR / "q4_building_type.json").write_text(
        json.dumps(q4_records, ensure_ascii=False),
        encoding="utf-8",
    )

    # ── q5: micro-districts ─────────────────────────────────────────────────
    q5 = res["q5"].reset_index().rename(columns={"Microdistrict": "microdistrict"})
    (OUTPUT_DIR / "q5_microdistricts.json").write_text(
        json.dumps([
            {
                "microdistrict": str(r["microdistrict"]),
                "avgPricePerM2": _clean(r["Avg_Price_per_m2"]),
                "listings": _clean(r["Listings"]),
            }
            for _, r in q5.iterrows()
        ], ensure_ascii=False),
        encoding="utf-8",
    )

    # ── q6: build year ──────────────────────────────────────────────────────
    q6 = res["q6"].reset_index().rename(columns={"Build_Year": "buildYear"})
    (OUTPUT_DIR / "q6_build_year.json").write_text(
        json.dumps([
            {
                "buildYear": int(r["buildYear"]),
                "medianPricePerM2": _clean(r["Median_Price_per_m2"]),
                "avgPricePerM2": _clean(r["Avg_Price_per_m2"]),
                "listings": _clean(r["Listings"]),
            }
            for _, r in q6.iterrows()
        ], ensure_ascii=False),
        encoding="utf-8",
    )

    # ── insights & questions ────────────────────────────────────────────────
    (OUTPUT_DIR / "insights.json").write_text(
        json.dumps(res["plain_language"], ensure_ascii=False),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "questions.json").write_text(
        json.dumps(ANALYTICAL_QUESTIONS, ensure_ascii=False),
        encoding="utf-8",
    )

    # ── listings (compact) ──────────────────────────────────────────────────
    cols_keep = [
        "Listing_ID", "Title", "Price_KZT", "Price_per_m2", "Area_m2", "Rooms",
        "District", "Address", "Build_Year", "Current_Floor", "Total_Floors",
        "Residential_Complex", "House_Type", "Ceiling_Height", "Bathroom",
        "Is_NewBuild", "Market_Segment", "Listing_URL",
    ]
    cols_keep = [c for c in cols_keep if c in df.columns]
    compact = df[cols_keep].copy()

    listings = []
    for _, r in compact.iterrows():
        rec = {}
        for c in cols_keep:
            rec[c] = _clean(r[c])
        listings.append(rec)
    (OUTPUT_DIR / "listings.json").write_text(
        json.dumps(listings, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Wrote {len(os.listdir(OUTPUT_DIR))} JSON files to {OUTPUT_DIR}")
    print(f"Listings exported: {len(listings)}")


if __name__ == "__main__":
    main()
