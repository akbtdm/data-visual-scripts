"""
Task 1: Web Scraper for krisha.kz — Almaty apartments (full catalog).

Applies Krisha's **year of construction** filter: `das[house.year][from]` … `[to]`
(default: last 2 calendar years through the current year, e.g. 2025–2026).
The year window is used **only as a search filter** on Krisha — it is no longer
exported as columns in the CSV.

`Is_NewBuild` is True only if `Listing_ID` is in that ID set (resales in new ЖК count as вторичка).
Legacy CSVs without `Listing_ID` may use text heuristics in preprocessing only.

Per-listing facts collected:
  * Build_Year           — Год постройки
  * Residential_Complex  — Жилой комплекс / ЖК
  * House_Type           — Тип дома (монолитный / кирпичный / панельный / …)
  * Ceiling_Height       — Высота потолков, метры (float)
  * Bathroom             — Санузел (совмещенный / раздельный / 2 санузла / …)
  * Developer            — Застройщик (только с детальной страницы)

Three passes:
  Pass A — collect novostroiki IDs (year-filtered).
  Pass B — full catalog (year-filtered); save raw CSV right after this pass so
           Pass C can never wipe progress.
  Pass C — per-listing detail pages (resumable, checkpointed every 100 rows).
           Skip rows where `Detail_Fetched=True`. Re-run with `--resume-details`
           to continue an interrupted enrichment.

CLI:
  python scraper.py                       # full pipeline
  python scraper.py --no-details          # skip Pass C (fastest)
  python scraper.py --resume-details      # only Pass C on an existing CSV
  python scraper.py --workers 16 --pages 3
"""

import time
import random
import re
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from typing import Optional, List, Dict, Any, Set

import requests
from bs4 import BeautifulSoup
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_URL = "https://krisha.kz/prodazha/kvartiry/almaty/"
NEWBUILDINGS_QUERY: Dict[str, Any] = {"das[novostroiki]": "1"}
# Default rolling window: `year_span` inclusive years ending in `year_end` (or today’s year).
DEFAULT_BUILD_YEAR_SPAN = 2
# When pages=None, stop after this many pages per pass (safety). Raise if you truly need more.
DEFAULT_MAX_PAGES_CAP = 10_000
# Concurrency for per-listing detail-page fetches (developer + fallback fields).
DEFAULT_DETAIL_WORKERS = 8
DEFAULT_DETAIL_DELAY_RANGE = (0.15, 0.45)
# Per-request timeout for detail pages: (connect, read). Krisha sometimes throttles slow
# responses, so a tight read timeout prevents one zombie listing from blocking a worker for 20 s.
DEFAULT_DETAIL_TIMEOUT: tuple = (5, 10)
# Pass C: log a progress line every N completed details (with rate + ETA).
DEFAULT_DETAIL_LOG_EVERY = 25
# Pass C: checkpoint to CSV every N completed details so Ctrl+C never wipes progress.
DEFAULT_DETAIL_CHECKPOINT_EVERY = 100
# Marker column added to every Pass-B record so Pass C is resumable.
DETAIL_DONE_COLUMN = "Detail_Fetched"
# Year boundaries we treat as a sane Almaty construction year (otherwise drop / mark None).
MIN_VALID_BUILD_YEAR = 1900
MAX_VALID_BUILD_YEAR = date.today().year + 5

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://krisha.kz/",
}


def _parse_price(text: str) -> Optional[int]:
    """Extract integer price from text like '36 500 000 〒'."""
    cleaned = re.sub(r"[^\d]", "", text.strip())
    return int(cleaned) if cleaned else None


def _parse_title(title: str) -> dict:
    """
    Parse '3-комнатная квартира · 85 м² · 5/9 этаж' into components.
    Returns dict with keys: Title, Rooms, Area_m2, Floor_Info.
    """
    result = {"Title": title.strip(), "Rooms": None, "Area_m2": None, "Floor_Info": None}

    rooms_match = re.search(r"(\d+)-комнатная", title)
    if rooms_match:
        result["Rooms"] = int(rooms_match.group(1))
    elif "студия" in title.lower():
        result["Rooms"] = 0

    area_match = re.search(r"([\d.,]+)\s*м²", title)
    if area_match:
        result["Area_m2"] = float(area_match.group(1).replace(",", "."))

    floor_match = re.search(r"(\d+)/(\d+)\s*этаж", title)
    if floor_match:
        result["Floor_Info"] = f"{floor_match.group(1)} из {floor_match.group(2)}"
    else:
        single_floor = re.search(r"(\d+)\s*этаж", title)
        if single_floor:
            result["Floor_Info"] = single_floor.group(1)

    return result


def _parse_location(subtitle: str) -> dict:
    """Split 'Бостандыкский р-н, мкр Орбита-4 6' into District + Address."""
    result = {"District": None, "Address": None}
    if not subtitle:
        return result
    parts = subtitle.split(",", 1)
    result["District"] = parts[0].strip()
    result["Address"] = parts[1].strip() if len(parts) > 1 else None
    return result


def _year_window_params(
    year_span: int = DEFAULT_BUILD_YEAR_SPAN,
    year_end: Optional[int] = None,
) -> Dict[str, str]:
    """
    Krisha URL params for «Год постройки»: only listings whose building year falls in [from, to].

    `year_span=5` and `year_end=2026` → 2022–2026 (five distinct calendar years).
    """
    if year_span < 1:
        raise ValueError("year_span must be >= 1")
    end = year_end if year_end is not None else date.today().year
    y_from = end - (year_span - 1)
    return {
        "das[house.year][from]": str(y_from),
        "das[house.year][to]": str(end),
    }


def _build_params(
    page: int,
    newbuildings_only: bool,
    year_params: Dict[str, str],
) -> Dict[str, Any]:
    params: Dict[str, Any] = dict(year_params)
    if newbuildings_only:
        params.update(NEWBUILDINGS_QUERY)
    if page > 1:
        params["page"] = page
    return params


def _fetch_soup(
    session: requests.Session,
    page: int,
    *,
    newbuildings_only: bool,
    year_params: Dict[str, str],
) -> Optional[BeautifulSoup]:
    params = _build_params(page, newbuildings_only, year_params)
    try:
        resp = session.get(BASE_URL, params=params, headers=HEADERS, timeout=20)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Fetch page %d (newbuild=%s) failed: %s", page, newbuildings_only, exc)
        return None
    return BeautifulSoup(resp.text, "lxml")


def _listing_ids_from_soup(soup: BeautifulSoup) -> List[str]:
    cards = soup.find_all("div", class_="a-card")
    return [str(c.get("data-id")) for c in cards if c.get("data-id")]


def infer_newbuild_from_text(description: str, title: str) -> bool:
    """
    Public helper: classify новостройка-style listing from snippet + title.
    Used by preprocessing when CSV has no scraper labels (legacy files).
    """
    return _infer_newbuild_heuristic(description or "", title or "")


def _infer_newbuild_heuristic(description: str, title: str) -> bool:
    """
    Fallback when listing ID is outside the sampled Krisha «новостройки» pages.
    Tuned for typical Krisha snippets (ЖК, year built, developer, etc.).
    """
    desc = (description or "").lower()
    blob = f"{desc} {(title or '').lower()}"

    if re.search(r"панельный дом,\s*19[0-8]\d", desc):
        return False
    if re.search(r"кирпичный дом,\s*19[0-6]\d", desc):
        return False

    if "от застройщика" in blob:
        return True
    if re.search(r"новостро", blob):
        return True

    if "жил. комплекс" in desc or re.search(r"\bжк\s", desc) or "жк «" in desc:
        years = [int(y) for y in re.findall(r"(20\d{2})\s*г\.п", desc)]
        if years and max(years) >= 2015:
            return True

    years = [int(y) for y in re.findall(r"(20[2-9]\d|201[6-9])\s*г\.п", desc)]
    if years and max(years) >= 2016:
        if "панельный дом, 19" not in desc:
            return True

    return False


def _classify_newbuild(listing_id: str, official_ids: Set[str]) -> bool:
    """
    True only if this listing ID appears in Krisha's «новостройки» search (Pass A).

    We intentionally do NOT use text heuristics here: resales in new ЖК still look like
    «жил. комплекс … г.п.» and would hide вторичка on charts. Heuristics are only for
    legacy CSVs in preprocessing when `Listing_ID` is missing.
    """
    if not listing_id:
        return False
    return str(listing_id) in official_ids


# ─────────────────────────────────────────────────────────────────────────────
# Per-listing facts: parse from card snippet first, then enrich from detail page.
# ─────────────────────────────────────────────────────────────────────────────

# canonical labels we try to recognise on Krisha (both card-snippet and detail page)
_HOUSE_TYPE_KEYWORDS = (
    "монолитный", "кирпичный", "панельный", "блочный",
    "каркасно-камышитовый", "каркасный", "иное",
)

_BATHROOM_KEYWORDS = (
    "2 с/у и более", "2 санузла", "санузел совмещенный", "санузел раздельный",
    "совмещенный", "раздельный",
)


def _to_float_meters(text: str) -> Optional[float]:
    """'3 м', '2.7м.', '2,75 м' → float meters."""
    if not text:
        return None
    m = re.search(r"([\d]+[.,]?\d*)", text.replace(",", "."))
    if not m:
        return None
    try:
        val = float(m.group(1))
    except ValueError:
        return None
    if 1.5 <= val <= 6.0:
        return round(val, 2)
    return None


def _valid_build_year(year: Optional[int]) -> Optional[int]:
    if year is None:
        return None
    if MIN_VALID_BUILD_YEAR <= year <= MAX_VALID_BUILD_YEAR:
        return year
    return None


def _parse_snippet_details(description: str) -> Dict[str, Any]:
    """
    Extract structured facts from Krisha card snippet, e.g.:
      'жил. комплекс Arena City. Park, 6 этажей, 2026 г.п., потолки 3м., санузел совмещенный, …'
    Returns dict; unknown values stay None.
    """
    out: Dict[str, Any] = {
        "Build_Year": None,
        "Residential_Complex": None,
        "House_Type": None,
        "Ceiling_Height": None,
        "Bathroom": None,
    }
    if not description:
        return out

    desc = description.strip()
    desc_lc = desc.lower()

    # Year of construction: '2026 г.п.' or '2024г.п'
    yr = re.search(r"(19\d{2}|20\d{2})\s*г\.?п", desc_lc)
    if yr:
        out["Build_Year"] = _valid_build_year(int(yr.group(1)))

    # Residential complex: 'жил. комплекс <Name>,' or 'ЖК «<Name>»' or 'ЖК <Name>,'
    rc = re.search(r"жил\.\s*комплекс\s+([^,]+?)(?=,)", desc, flags=re.IGNORECASE)
    if not rc:
        rc = re.search(r"ЖК\s+«([^»]+)»", desc)
    if not rc:
        rc = re.search(r"\bЖК\s+([^,]+?)(?=,)", desc)
    if rc:
        name = rc.group(1).strip().strip("«»\"' ")
        if name:
            out["Residential_Complex"] = name

    # House type — only if the snippet explicitly says e.g. 'монолитный дом'
    ht_match = re.search(r"\b(монолитн\w+|кирпичн\w+|панельн\w+|блочн\w+|каркасн\w+)\s+дом", desc_lc)
    if ht_match:
        token = ht_match.group(1)
        for canon in _HOUSE_TYPE_KEYWORDS:
            if canon.startswith(token[:6]):
                out["House_Type"] = canon
                break

    # Ceiling height: 'потолки 3м.' / 'потолки 2.7 м'
    ch = re.search(r"потолк\w*\s*([\d.,]+)\s*м", desc_lc)
    if ch:
        out["Ceiling_Height"] = _to_float_meters(ch.group(1))

    # Bathroom: 'санузел совмещенный', 'санузел раздельный', '2 с/у'
    if re.search(r"\b2\s*с/у\b|\b2\s*санузла", desc_lc):
        out["Bathroom"] = "2 санузла и более"
    else:
        b = re.search(r"санузел\s+(совмещ\w+|раздельн\w+)", desc_lc)
        if b:
            out["Bathroom"] = "совмещенный" if b.group(1).startswith("совмещ") else "раздельный"

    return out


# Mapping from Krisha "info-title" labels (lowercased, stripped) to our column names.
_DETAIL_LABEL_MAP = {
    "тип дома": "House_Type",
    "жилой комплекс": "Residential_Complex",
    "год постройки": "Build_Year",
    "санузел": "Bathroom",
    "высота потолков": "Ceiling_Height",
    "застройщик": "Developer",
}


def _coerce_detail_value(field: str, raw: str) -> Any:
    """Map raw Krisha label-text to typed Python value for the given column."""
    text = raw.strip()
    if not text:
        return None
    if field == "Build_Year":
        m = re.search(r"(19\d{2}|20\d{2})", text)
        return _valid_build_year(int(m.group(1))) if m else None
    if field == "Ceiling_Height":
        return _to_float_meters(text)
    if field == "House_Type":
        low = text.lower()
        for canon in _HOUSE_TYPE_KEYWORDS:
            if canon in low:
                return canon
        return text
    if field == "Bathroom":
        low = text.lower()
        if "2" in low and ("с/у" in low or "санузл" in low):
            return "2 санузла и более"
        if "совмещ" in low:
            return "совмещенный"
        if "раздельн" in low:
            return "раздельный"
        return text
    return text  # Residential_Complex, Developer — keep verbatim


def _parse_detail_page(soup: BeautifulSoup) -> Dict[str, Any]:
    """
    Pull labelled facts from a Krisha listing detail page.
    Krisha uses `<div class="offer__info-item">` blocks containing
    `<div class="offer__info-title">…</div><div class="offer__info-value">…</div>`.
    Falls back to legacy `<dl class="offer__parameters"><dt>/<dd>` layout if needed.
    """
    facts: Dict[str, Any] = {k: None for k in _DETAIL_LABEL_MAP.values()}

    items = soup.find_all("div", class_="offer__info-item")
    for item in items:
        title_el = item.find(class_="offer__info-title")
        value_el = item.find(class_="offer__info-value")
        if not title_el or not value_el:
            continue
        label = title_el.get_text(strip=True).lower()
        field = _DETAIL_LABEL_MAP.get(label)
        if not field:
            continue
        value = _coerce_detail_value(field, value_el.get_text(" ", strip=True))
        if value is not None and facts.get(field) in (None, ""):
            facts[field] = value

    # Legacy `<dl>` layout fallback.
    if not any(facts.values()):
        for dl in soup.find_all("dl", class_="offer__parameters"):
            for dt, dd in zip(dl.find_all("dt"), dl.find_all("dd")):
                label = dt.get_text(strip=True).lower()
                field = _DETAIL_LABEL_MAP.get(label)
                if not field:
                    continue
                value = _coerce_detail_value(field, dd.get_text(" ", strip=True))
                if value is not None and facts.get(field) in (None, ""):
                    facts[field] = value

    # Developer block: <div class="owners__name">Застройщик BI Group Юг</div>.
    # Resales show the seller's name in the same slot, so accept only entries that
    # explicitly start with "Застройщик".
    if not facts.get("Developer"):
        for owner_el in soup.find_all(class_=re.compile(r"owners__name|builder__name|developer__title")):
            text = owner_el.get_text(" ", strip=True)
            if not text:
                continue
            stripped = re.sub(r"^застройщик\s+", "", text, flags=re.IGNORECASE).strip()
            if stripped and stripped.lower() != text.lower():
                # Prefix "Застройщик " was present → it's a real developer label.
                facts["Developer"] = stripped
                break

    return facts


def _fetch_listing_details(
    session: requests.Session,
    url: str,
    *,
    timeout: tuple = DEFAULT_DETAIL_TIMEOUT,
) -> Dict[str, Any]:
    """Fetch one listing detail page and return parsed facts (empty dict on error)."""
    try:
        resp = session.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.debug("Detail fetch failed for %s: %s", url, exc)
        return {}
    soup = BeautifulSoup(resp.text, "lxml")
    return _parse_detail_page(soup)


def _enrich_with_details(
    records: List[dict],
    *,
    workers: int = DEFAULT_DETAIL_WORKERS,
    delay_range: tuple = DEFAULT_DETAIL_DELAY_RANGE,
    request_timeout: tuple = DEFAULT_DETAIL_TIMEOUT,
    checkpoint_path: Optional[str] = None,
    checkpoint_every: int = DEFAULT_DETAIL_CHECKPOINT_EVERY,
    log_every: int = DEFAULT_DETAIL_LOG_EVERY,
) -> None:
    """
    For each record fetch its detail page in parallel and fill in any missing
    Build_Year / Residential_Complex / House_Type / Ceiling_Height / Bathroom / Developer.
    Modifies records in place.

    Resumable: skips any record where ``Detail_Fetched`` is truthy. After a successful
    (or failed) fetch, the record is marked ``Detail_Fetched=True`` so a re-run only
    processes outstanding rows.

    If ``checkpoint_path`` is provided, the full ``records`` list is dumped to that CSV
    every ``checkpoint_every`` completions (and once at the end). This means Ctrl+C
    never wipes progress — at most you lose ``checkpoint_every`` last fetches.
    """
    if not records:
        return

    pending = [r for r in records if not r.get(DETAIL_DONE_COLUMN)]
    already = len(records) - len(pending)
    if not pending:
        logger.info(
            "All %d records already have %s=True — nothing to enrich.",
            len(records), DETAIL_DONE_COLUMN,
        )
        return

    logger.info(
        "Pass C: enriching %d/%d records (already done %d, workers=%d, timeout=%s, "
        "checkpoint every %d, log every %d)…",
        len(pending), len(records), already,
        workers, request_timeout, checkpoint_every, log_every,
    )

    session = requests.Session()
    completed = 0
    start = time.monotonic()

    def _save_checkpoint(tag: str) -> None:
        if not checkpoint_path:
            return
        pd.DataFrame(records).to_csv(checkpoint_path, index=False, encoding="utf-8-sig")
        logger.info("  checkpoint (%s) → %s", tag, checkpoint_path)

    def worker(rec: dict) -> tuple:
        time.sleep(random.uniform(*delay_range))
        url = rec.get("Listing_URL")
        if not url:
            return rec, {}
        return rec, _fetch_listing_details(session, url, timeout=request_timeout)

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(worker, rec) for rec in pending]
            for fut in as_completed(futures):
                try:
                    rec, detail = fut.result()
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Detail worker error: %s", exc)
                    completed += 1
                    continue
                for field, value in detail.items():
                    if value is None:
                        continue
                    if rec.get(field) in (None, ""):
                        rec[field] = value
                rec[DETAIL_DONE_COLUMN] = True
                completed += 1

                if completed % log_every == 0 or completed == len(pending):
                    elapsed = max(time.monotonic() - start, 1e-6)
                    rate = completed / elapsed
                    remaining = len(pending) - completed
                    eta_min = remaining / rate / 60 if rate > 0 else float("inf")
                    logger.info(
                        "  details %d/%d (%.2f/s, ETA %.1f min)",
                        completed, len(pending), rate, eta_min,
                    )

                if checkpoint_path and checkpoint_every > 0 and completed % checkpoint_every == 0:
                    _save_checkpoint(f"{completed}/{len(pending)}")
    finally:
        _save_checkpoint("final")


def enrich_csv_with_details(
    csv_path: str = "krisha_raw.csv",
    *,
    workers: int = DEFAULT_DETAIL_WORKERS,
    request_timeout: tuple = DEFAULT_DETAIL_TIMEOUT,
    checkpoint_every: int = DEFAULT_DETAIL_CHECKPOINT_EVERY,
    log_every: int = DEFAULT_DETAIL_LOG_EVERY,
) -> pd.DataFrame:
    """
    Load an existing raw CSV and run Pass C (detail enrichment) on rows where
    ``Detail_Fetched`` is False/missing. Saves checkpoints back to the same CSV.

    Use this to *resume* an interrupted detail enrichment without re-doing Pass A+B.
    """
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if DETAIL_DONE_COLUMN not in df.columns:
        df[DETAIL_DONE_COLUMN] = False
    df[DETAIL_DONE_COLUMN] = (
        df[DETAIL_DONE_COLUMN]
        .map(lambda x: bool(x) if not pd.isna(x) else False)
        .astype(bool)
    )

    records = df.to_dict(orient="records")
    _enrich_with_details(
        records,
        workers=workers,
        request_timeout=request_timeout,
        checkpoint_path=csv_path,
        checkpoint_every=checkpoint_every,
        log_every=log_every,
    )
    return pd.DataFrame(records)


def _parse_cards_from_soup(soup: BeautifulSoup, official_newbuild_ids: Set[str]) -> List[dict]:
    cards = soup.find_all("div", class_="a-card")
    records = []
    for card in cards:
        if not card.get("data-id"):
            continue

        listing_id = str(card.get("data-id"))
        title_tag = card.find("a", class_="a-card__title")
        price_tag = card.find("div", class_="a-card__price")
        subtitle_tag = card.find("div", class_="a-card__subtitle")
        desc_tag = card.find("div", class_="a-card__text-preview")

        if not title_tag or not price_tag:
            continue

        raw_title = title_tag.get_text(strip=True)
        raw_price = price_tag.get_text(strip=True)
        raw_subtitle = subtitle_tag.get_text(strip=True) if subtitle_tag else ""
        raw_desc = desc_tag.get_text(strip=True) if desc_tag else ""

        parsed_title = _parse_title(raw_title)
        parsed_loc = _parse_location(raw_subtitle)
        desc_snippet = raw_desc[:300] if raw_desc else None
        snippet_facts = _parse_snippet_details(raw_desc)

        is_new = _classify_newbuild(listing_id, official_newbuild_ids)

        record = {
            "Listing_ID": listing_id,
            "Is_NewBuild": is_new,
            "Title": parsed_title["Title"],
            "Price_KZT": _parse_price(raw_price),
            "District": parsed_loc["District"],
            "Address": parsed_loc["Address"],
            "Floor_Info": parsed_title["Floor_Info"],
            "Area_m2": parsed_title["Area_m2"],
            "Rooms": parsed_title["Rooms"],
            "Build_Year": snippet_facts["Build_Year"],
            "Residential_Complex": snippet_facts["Residential_Complex"],
            "House_Type": snippet_facts["House_Type"],
            "Ceiling_Height": snippet_facts["Ceiling_Height"],
            "Bathroom": snippet_facts["Bathroom"],
            "Developer": None,
            "Description_Snippet": desc_snippet,
            "Listing_URL": "https://krisha.kz" + title_tag.get("href", ""),
            DETAIL_DONE_COLUMN: False,
        }
        records.append(record)

    return records


def scrape(
    pages: Optional[int] = None,
    output_path: str = "krisha_raw.csv",
    *,
    max_pages_cap: int = DEFAULT_MAX_PAGES_CAP,
    year_span: int = DEFAULT_BUILD_YEAR_SPAN,
    year_end: Optional[int] = None,
    fetch_details: bool = True,
    detail_workers: int = DEFAULT_DETAIL_WORKERS,
    max_listings: Optional[int] = None,
) -> pd.DataFrame:
    """
    Two-pass scrape with Krisha **construction-year** filter on every request.

      Pass A — das[novostroiki]=1 + year window: collect official new-build IDs.
      Pass B — full segment (year-filtered); Is_NewBuild = membership in Pass A ID set only.
      Pass C — (optional, default ON) per-listing detail pages → Developer + missing facts.

    The year window is applied to Krisha URLs only (search filter); it is **not**
    persisted as columns in the CSV.

    Args:
        pages: If ``int`` — fixed number of pages per pass (legacy). If ``None`` — fetch **all**
            pages until Krisha returns 0 listings (or ``max_pages_cap`` is reached).
        max_pages_cap: Hard limit per pass when ``pages is None`` (avoids infinite loops).
        output_path: Raw CSV path.
        year_span: Number of consecutive calendar years to include (default 5).
        year_end: Last year of the window (default: current calendar year).
        fetch_details: If True (default), fetch each listing's detail page in parallel
            to populate ``Developer`` and any snippet-missing facts.
        detail_workers: Number of parallel workers for detail-page fetches.

    Returns:
        DataFrame with per-listing facts (no filter-window columns).
    """
    year_params = _year_window_params(year_span, year_end)
    y_from = int(year_params["das[house.year][from]"])
    y_to = int(year_params["das[house.year][to]"])
    logger.info(
        "Krisha filter: year of construction %d–%d (%d-year window, year_end=%d)",
        y_from, y_to, year_span, y_to,
    )

    if pages is not None:
        logger.info("Pagination: fixed %d page(s) per pass", pages)
        page_iter_a = range(1, pages + 1)
        page_iter_b = range(1, pages + 1)
        auto_stop = False
    else:
        logger.info(
            "Pagination: ALL pages until empty (max %d per pass, ~1–2.5s between requests)",
            max_pages_cap,
        )
        page_iter_a = range(1, max_pages_cap + 1)
        page_iter_b = range(1, max_pages_cap + 1)
        auto_stop = True

    session = requests.Session()
    official_ids: Set[str] = set()

    logger.info("Pass A: new-build IDs (novostroiki + year)...")
    for page in page_iter_a:
        soup = _fetch_soup(session, page, newbuildings_only=True, year_params=year_params)
        if soup is None:
            logger.warning("Pass A: stopping at page %d (request failed)", page)
            break
        page_ids = _listing_ids_from_soup(soup)
        if not page_ids:
            if auto_stop:
                logger.info("Pass A: finished at page %d (0 listings — end of results)", page)
                break
            logger.info("  Page %d → +0 IDs", page)
        else:
            official_ids.update(page_ids)
            logger.info("  Page %d → +%d IDs (total unique %d)", page, len(page_ids), len(official_ids))
        if auto_stop and page >= max_pages_cap:
            if page_ids:
                logger.warning(
                    "Pass A: hit max_pages_cap=%d but still got listings — raise max_pages_cap for a complete crawl",
                    max_pages_cap,
                )
            break
        time.sleep(random.uniform(1.0, 2.5))

    if max_listings is not None and max_listings > 0:
        logger.info("Pass B will stop early once %d listings are collected (max_listings cap)", max_listings)

    logger.info("Pass B: full catalog (year-filtered only)...")
    all_records: List[dict] = []
    for page in page_iter_b:
        soup = _fetch_soup(session, page, newbuildings_only=False, year_params=year_params)
        if soup is None:
            logger.warning("Pass B: stopping at page %d (request failed)", page)
            break
        recs = _parse_cards_from_soup(soup, official_ids)
        if not recs:
            if auto_stop:
                logger.info("Pass B: finished at page %d (0 listings — end of results)", page)
                break
            logger.info("Page %d → 0 listings", page)
        else:
            all_records.extend(recs)
            logger.info("Page %d → %d listings (cumulative %d)", page, len(recs), len(all_records))
        if max_listings is not None and len(all_records) >= max_listings:
            all_records = all_records[:max_listings]
            logger.info(
                "Pass B: hit max_listings cap %d at page %d — stopping early",
                max_listings, page,
            )
            break
        if auto_stop and page >= max_pages_cap:
            if recs:
                logger.warning(
                    "Pass B: hit max_pages_cap=%d but still got listings — raise max_pages_cap for a complete crawl",
                    max_pages_cap,
                )
            break
        delay = random.uniform(1.0, 2.5)
        logger.info("Sleeping %.1fs before next page...", delay)
        time.sleep(delay)

    # ── Save Pass A+B output BEFORE Pass C ───────────────────────────────────
    # This way Ctrl+C during Pass C (or a crash) never destroys ~25 min of crawling.
    # Pass C is then resumable: it will only fetch rows where Detail_Fetched is False.
    df = pd.DataFrame(all_records)
    if not df.empty:
        nb = int(df["Is_NewBuild"].sum())
        logger.info("Labels: %d new-build (%.1f%%), %d secondary", nb, 100 * nb / len(df), len(df) - nb)

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info("Saved %d raw rows (pre-details) to '%s'", len(df), output_path)

    # ── Pass C: per-listing detail enrichment (resumable, checkpointed) ───────
    if fetch_details and all_records:
        _enrich_with_details(
            all_records,
            workers=detail_workers,
            checkpoint_path=output_path,
        )
        df = pd.DataFrame(all_records)

    if not df.empty:
        for col in ("Build_Year", "Developer", "Residential_Complex",
                    "House_Type", "Ceiling_Height", "Bathroom"):
            if col in df.columns:
                filled = int(df[col].notna().sum())
                logger.info("  %s populated: %d/%d (%.1f%%)",
                            col, filled, len(df), 100 * filled / len(df))

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Krisha.kz scraper — Pass A (новостройки IDs) + Pass B (catalog) + Pass C (details).",
    )
    parser.add_argument(
        "--resume-details", action="store_true",
        help="Skip Pass A+B; just resume Pass C (detail enrichment) for existing CSV.",
    )
    parser.add_argument(
        "--no-details", action="store_true",
        help="Skip Pass C entirely (raw CSV ready in ~25 min, Developer left blank).",
    )
    parser.add_argument(
        "--pages", type=int, default=None,
        help="Fixed page count per pass (default: crawl until empty).",
    )
    parser.add_argument(
        "--year-span", type=int, default=DEFAULT_BUILD_YEAR_SPAN,
        help=f"Construction-year window length (default: {DEFAULT_BUILD_YEAR_SPAN}).",
    )
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_DETAIL_WORKERS,
        help=f"Parallel workers for Pass C (default: {DEFAULT_DETAIL_WORKERS}).",
    )
    parser.add_argument(
        "--max-listings", type=int, default=None,
        help="Stop Pass B once this many listings are collected (cap on total dataset size).",
    )
    parser.add_argument(
        "--output", default="krisha_raw.csv",
        help="Output CSV path (default: krisha_raw.csv).",
    )
    args = parser.parse_args()

    if args.resume_details:
        logger.info("Resume mode: enriching '%s' (workers=%d)…", args.output, args.workers)
        df = enrich_csv_with_details(args.output, workers=args.workers)
    else:
        df = scrape(
            pages=args.pages,
            output_path=args.output,
            year_span=args.year_span,
            fetch_details=not args.no_details,
            detail_workers=args.workers,
            max_listings=args.max_listings,
        )
    print(df.head())
    print(f"\nTotal rows: {len(df)}")
    if "Is_NewBuild" in df.columns:
        print(df["Is_NewBuild"].value_counts())
    extra_cols = ["Build_Year", "Developer", "Residential_Complex",
                  "House_Type", "Ceiling_Height", "Bathroom", DETAIL_DONE_COLUMN]
    present = [c for c in extra_cols if c in df.columns]
    if present:
        print("\n=== Sample of detail fields ===")
        print(df[["Listing_ID"] + present].head(10).to_string(index=False))
