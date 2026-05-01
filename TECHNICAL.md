# Technical Documentation — Almaty Housing Market (Krisha.kz)

INF 369 Final Project: end-to-end pipeline for scraping, cleaning, analyzing, and visualizing apartment listings in Almaty.

---

## 1. Overview

The project implements five tasks in modular Python:

| Task | Module | Output |
|------|--------|--------|
| Web scraping | `scraper.py` | `krisha_raw.csv` (full catalog + `Listing_ID`, `Is_NewBuild`) |
| Preprocessing | `preprocessing.py` | `krisha_cleaned.csv` |
| Analysis | `analysis.py` | In-memory results + logs |
| Visualization | `visualization.py` | `charts/` — **10** figures (1–5 market, 6–10 new vs secondary) |
| Orchestration + report | `main.py` | Console summary, `pipeline.log` |

The orchestrator runs tasks in order: **Scrape → Clean → Analyze → Visualize → Storytelling report**.

---

## 2. Environment, tools & stack

### 2.1 Runtime prerequisites

| Resource | Requirement |
|----------|-------------|
| **Python** | 3.9+ (project uses `typing.Optional` / `Tuple` for 3.9; avoid `list[str]` union syntax on older runtimes). |
| **Network** | HTTPS to `krisha.kz` for scraping. |
| **Disk** | ~2–5 MB per CSV; Plotly HTML scatter can be ~5 MB. |
| **Display** | Not required: Matplotlib uses the `Agg` backend (no GUI). |

Install third-party libraries (pinned minimums in `requirements.txt`):

```bash
pip3 install -r requirements.txt
```

To record exact versions for a reproducible environment (recommended for submissions):

```bash
pip3 freeze > requirements-lock.txt
```

### 2.2 Stack at a glance

| Layer | Tools |
|-------|--------|
| Language | Python 3 |
| HTTP client | `requests` |
| HTML parsing | `beautifulsoup4` + `lxml` |
| Tabular data | `pandas` (built on `numpy`) |
| Statistics | `scipy` (via `pandas` for Spearman) |
| Static plots | `matplotlib` + `seaborn` |
| Interactive plot | `plotly` (HTML export) |
| Orchestration | `logging`, `pathlib`, `random`, `re`, `time` (stdlib) |

### 2.3 Third-party libraries (what each does & where)

**requests** (`>=2.31.0`)

- **Role:** HTTP/HTTPS client with session support, timeouts, and simple header control.
- **Used in:** `scraper.py` — `requests.Session()`, `GET` with `params={"page": n}`, custom `User-Agent` / `Referer`.
- **Why not `urllib` alone:** Cleaner API for headers, sessions, and status handling; assignment-friendly for INF 369 “requests + BeautifulSoup” requirement.

**beautifulsoup4** (`>=4.12.0`)

- **Role:** Parse HTML/XML into a navigable tree; CSS-friendly `find` / `find_all`.
- **Used in:** `scraper.py` — locate `div.a-card`, `a.a-card__title`, `div.a-card__price`, etc.
- **Parser:** Passed to BeautifulSoup as `"lxml"` for speed and tolerance of imperfect markup.

**lxml** (`>=5.0.0`)

- **Role:** Fast C-backed parser and tree engine for BeautifulSoup.
- **Used in:** `BeautifulSoup(html, "lxml")` in `scraper.py`.
- **Note:** Falls back to `html.parser` only if you change the code; `lxml` is the supported path in this project.

**pandas** (`>=2.1.0`)

- **Role:** `DataFrame` I/O (`read_csv` / `to_csv`), cleaning, `groupby`, correlations, nullable integers (`Int64`).
- **Used in:** All pipeline stages after scrape — `preprocessing.py`, `analysis.py`, `visualization.py`, `main.py`.

**numpy** (`>=1.26.0`)

- **Role:** Numeric arrays; underpins pandas and plotting helpers.
- **Used in:** Implicitly via pandas; `numpy` imported in `visualization.py` for heatmap mask / array ops.

**scipy** (`>=1.11.0`)

- **Role:** Scientific computing; **Spearman correlation** is computed through `pandas.Series.corr(..., method="spearman")`, which delegates to `scipy.stats` when available.
- **Used in:** `analysis.py` — Q2 (rooms vs. price per m²).
- **If missing:** `ModuleNotFoundError` on Spearman path; install `scipy` or switch Q2 to Pearson-only.

**matplotlib** (`>=3.8.0`)

- **Role:** Low-level plotting, figures, axes, file export (`savefig`).
- **Used in:** `visualization.py` — box plot, scatter (static), heatmap, bar chart, histogram; `matplotlib.use("Agg")` for servers/CI.

**seaborn** (`>=0.13.0`)

- **Role:** Statistical visuals on top of Matplotlib (boxplot, heatmap styling, color palettes).
- **Used in:** `visualization.py` — district box plots, heatmap, scatter colors.

**plotly** (`>=5.18.0`)

- **Role:** Interactive charts exported as self-contained HTML (pan/zoom, hover tooltips).
- **Used in:** `visualization.py` — `plotly.express.scatter` → `charts/2_scatter_area_vs_price.html`.
- **Note:** OLS trendlines were omitted to avoid pulling in `statsmodels`; static Matplotlib scatter still ships as PNG.

### 2.4 Python standard library (built-in tools)

| Module | Role in this project |
|--------|----------------------|
| `logging` | INFO-level pipeline trace; file + console in `main.py`; module loggers elsewhere. |
| `pathlib.Path` | Check `krisha_raw.csv` existence; chart output directory in `visualization.py`. |
| `re` | Title/price/floor parsing in `scraper.py` and `preprocessing.py`. |
| `random` | `uniform` sleep between scrape pages. |
| `time` | `sleep()` in scraper between requests. |
| `sys` | Stream handler for logging in `main.py`. |
| `warnings` | Suppress noisy third-party warnings in `visualization.py`. |
| `typing` | `Optional`, `Tuple`, `List` for Python 3.9 type hints. |

### 2.5 Development & delivery tools (outside `requirements.txt`)

| Tool | How it helps |
|------|----------------|
| **Git** | Version control for code and docs; ignore `*.csv`, `charts/`, `pipeline.log` if you do not want binaries in repo. |
| **venv / conda** | Isolate `pip` packages so course machine matches your laptop. |
| **Cursor / VS Code** | Edit-run loop; project rule `.cursor/rules/run-python-check.mdc` nudges “run Python after edits.” |
| **Browser** | Open `charts/2_scatter_area_vs_price.html` for interactive exploration; PNGs for slides/PDF. |
| **Jupyter / Quarto** | Optional: import `pandas` and `analyze()` for narrative notebooks — not required by the current repo layout. |

### 2.6 Tooling choices vs. alternatives (for your report)

- **BeautifulSoup + requests** matches the course “Option A” scraping stack and handles Krisha’s server-rendered list HTML without a headless browser.
- **pandas** is the standard for tabular EDA in data science courses; same code patterns extend to SQL/DuckDB later.
- **Seaborn + Matplotlib** satisfy static, publication-style figures; **Plotly** adds one “wow” interactive artifact without running a dashboard server.

---

## 3. Quick start

From the project root (`data_final/`):

```bash
python3 main.py
```

- If `krisha_raw.csv` **exists**, scraping is **skipped** (faster re-runs for cleaning/analysis).
- To force a full re-scrape, delete `krisha_raw.csv` first.

---

## 4. Repository layout

```
data_final/
├── main.py              # Entry point
├── scraper.py           # Task 1
├── preprocessing.py     # Task 2
├── analysis.py          # Task 3
├── visualization.py     # Task 4
├── requirements.txt
├── TECHNICAL.md         # This file
├── krisha_raw.csv       # Generated (gitignore recommended)
├── krisha_cleaned.csv   # Generated
├── pipeline.log         # Generated
└── charts/              # Generated figures
```

---

## 5. Pipeline architecture

```text
┌─────────────┐     ┌─────────────────┐     ┌──────────────┐
│  scraper.py │ ──► │ preprocessing.py │ ──► │ analysis.py  │
│  (HTTP+BS4) │     │  (pandas clean)  │     │  (pandas Qs) │
└─────────────┘     └─────────────────┘     └──────┬───────┘
       │                      │                     │
       ▼                      ▼                     ▼
 krisha_raw.csv      krisha_cleaned.csv      results dict
                            │                     │
                            └──────────┬──────────┘
                                       ▼
                              visualization.py
                                       │
                                       ▼
                                  charts/
                                       │
                                       ▼
                              main.py: print_summary_report()
```

---

## 6. Module reference

### 6.1 `scraper.py`

**Purpose:** Download *all* apartment listings for Almaty, label **новостройка vs вторичка**, and parse cards.

**Two-pass HTTP (same `pages` for each pass)**

1. **Pass A — ID harvest:** For each page `1…pages`, request with `das[novostroiki]=1` (Krisha checkbox «новостройки»). Collect every `data-id` into a set `official_ids`.
2. **Pass B — Full catalog:** Same page range **without** the filter. `Is_NewBuild = (Listing_ID ∈ official_ids)` only — resales in new ЖК stay **вторичка** (heuristic is not applied in the scraper).

**Heuristic** (when an ID is not in the sampled new-build pages): developer phrases, «жил. комплекс» / ЖК + recent `г.п.` year, «новостро», etc.; old Soviet panel/brick patterns → secondary.

**HTTP**

- Base URL: `https://krisha.kz/prodazha/kvartiry/almaty/`
- **Year of construction (default 5-year window):** `das[house.year][from]` and `das[house.year][to]` — e.g. in 2026 with `year_span=5` → `2022`…`2026`. Applied on **both** Pass A and Pass B.
- Pagination: `page=N` (N ≥ 2), combined with `das[novostroiki]=1` only in Pass A.
- Delay: random **1.0–2.5 s** between **every** request (both passes).

**DOM selectors**

- Card root: `div.a-card` with `data-id` → `Listing_ID`.
- Title: `a.a-card__title`; Price: `div.a-card__price`; Subtitle: `div.a-card__subtitle`; Snippet: `div.a-card__text-preview`.

**Public API**

```python
scrape(
    pages: Optional[int] = None,
    output_path: str = "krisha_raw.csv",
    *,
    max_pages_cap: int = 10_000,
    year_span: int = 5,
    year_end: Optional[int] = None,
) -> pd.DataFrame
```

- **`pages=None` (default in `main.py`):** keep requesting the next page until Krisha returns **0** listings, separately for Pass A and Pass B (still bounded by **`max_pages_cap`** per pass).
- **`pages=30`:** legacy fixed depth (~600 rows per pass if every page is full).

**CLI:** `python3 scraper.py` — quick test (3 pages); full crawl: `scrape(pages=None)` (can take **hours** if the filtered catalog has thousands of pages — ~1–2.5 s delay × 2 passes × pages).

---

### 6.2 `preprocessing.py`

**Purpose:** Turn raw rows into analysis-ready tabular data.

**Steps (order matters)**

0. `Is_NewBuild`: CSV-safe booleans. **Legacy CSV** without `Listing_ID` and all `False`: optional text heuristic. If `Listing_ID` is present, **never** override scraper labels (even when all `False` = всё не из выдачи «новостройки»).
1. Parse `Floor_Info` → `Current_Floor`, `Total_Floors` (supports `N из M`, `N/M`, single floor).
2. Coerce `Price_KZT`, `Area_m2`, `Rooms` to numeric; drop rows missing price or area.
3. Drop duplicates by `Listing_URL`, then by `(Price_KZT, Area_m2, Address)`.
4. Remove listings with `Area_m2 < 10`, `Price_KZT > 1_000_000_000`, or non-positive price.
5. Trim extreme prices: keep rows between **1st and 99th percentile** of `Price_KZT`.
6. `Price_per_m2 = Price_KZT / Area_m2` (rounded to whole tenge).
7. **District normalization:** map subtitle prefixes to canonical names (`…ский р-н`); unmatched → `Прочее`.
8. Fill: `Rooms` (median), `Address`, `Description_Snippet`; floors remain nullable. `Is_NewBuild` is preserved through row drops.

**Public API**

```python
clean(input_path: str = "krisha_raw.csv", output_path: str = "krisha_cleaned.csv") -> pd.DataFrame
```

---

### 6.3 `analysis.py`

**Purpose:** Answer five analytical questions; return a `dict` with keys `q1`–`q5`.

| Key | Question | Method |
|-----|----------|--------|
| `q1` | Average ₸/m² by district | `groupby("District")` on `Price_per_m2` |
| `q2` | Rooms vs ₸/m² correlation | Pearson + Spearman; breakdown by room count |
| `q3` | First / Middle / Last floor prices | Derived `Floor_Level` from `Current_Floor` vs `Total_Floors` |
| `q4` | Khrushchevka vs high-rise | `Total_Floors ≤ 5`, `6–11`, `≥ 12` building buckets |
| `q5` | Top micro-districts | First segment of `Address` as `Microdistrict`; min 3 listings; top by mean ₸/m² |

**Public API**

```python
analyze(input_path: str = "krisha_cleaned.csv") -> dict
```

---

### 6.4 `visualization.py`

**Purpose:** Seaborn/Matplotlib static figures + Plotly HTML scatter plots.

| # | File | Description |
|---|------|-------------|
| 1 | `charts/1_boxplot_price_by_district.png` | Box plot of `Price_KZT` by district (outliers hidden for readability) |
| 2 | `charts/2_scatter_area_vs_price.png` + `.html` | Area vs price, color = rooms |
| 3 | `charts/3_heatmap_correlation.png` | Correlation matrix: price, area, rooms, floors, ₸/m² |
| 4 | `charts/4_bar_avg_price_per_m2_district.png` | Horizontal bar, districts sorted by mean ₸/m² |
| 5 | `charts/5_histogram_price_per_m2.png` | Distribution of ₸/m²; highlights IQR band |
| 6 | `charts/6_newbuild_bar_mean_price.png` | Mean `Price_KZT` & mean `Price_per_m2` — **Новостройка vs Вторичка** |
| 7 | `charts/7_newbuild_boxplot_price_per_m2.png` | Box plot `Price_per_m2` by segment |
| 8 | `charts/8_newbuild_scatter_area_vs_price.png` + `.html` | Area vs price, color = segment |
| 9 | `charts/9_newbuild_stacked_counts_by_district.png` | Stacked counts (top 8 districts × segment) |
| 10 | `charts/10_newbuild_hist_price_per_m2_overlay.png` | Overlapping normalized histograms of ₸/m² |

Uses helper column `Segment` (`Новостройка` / `Вторичка`) derived from `Is_NewBuild`.

Matplotlib uses non-interactive backend `Agg` for headless runs.

**Public API**

```python
visualize(input_path: str = "krisha_cleaned.csv") -> None
```

---

### 6.5 `main.py`

**Purpose:** Wire modules, logging, and **Task 5** storytelling (`print_summary_report`).

**Constants**

- `SCRAPE_PAGES = 30` (~600 listings; adjust if site layout changes).
- `RAW_CSV`, `CLEAN_CSV` — paths relative to CWD.

**Logging:** stdout + `pipeline.log` (UTF-8).

---

## 7. Data schemas

### 7.1 `krisha_raw.csv`

| Column | Type | Notes |
|--------|------|-------|
| Listing_ID | str | Krisha `data-id` |
| Is_NewBuild | bool | `True` iff `Listing_ID` ∈ Pass A «новостройки» set (no scraper heuristic) |
| Title | str | Full listing title |
| Price_KZT | int | Parsed integer |
| District | str | First segment of subtitle (pre-normalization) |
| Address | str | Rest of subtitle |
| Floor_Info | str | e.g. `5 из 9` or single floor |
| Area_m2 | float | From title |
| Rooms | int | From title |
| Description_Snippet | str | Truncated preview |
| Listing_URL | str | Canonical link |
| Filter_Year_From | int | Krisha `house.year` lower bound used when scraping |
| Filter_Year_To | int | Krisha `house.year` upper bound |

### 7.2 `krisha_cleaned.csv`

All raw columns above (except district may be canonicalized) **plus:**

| Column | Type |
|--------|------|
| Current_Floor | Int64 (nullable) |
| Total_Floors | Int64 (nullable) |
| Price_per_m2 | float |

---

## 8. Configuration knobs

| Location | Parameter | Effect |
|----------|-----------|--------|
| `main.py` | `SCRAPE_PAGES` | `None` = all pages until empty; or an `int` fixed page count per pass |
| `main.py` | `SCRAPE_MAX_PAGES_CAP` | Hard stop per pass when `SCRAPE_PAGES is None` (raise if Krisha needs more) |
| `main.py` | `BUILD_YEAR_SPAN` | Construction-year window → `scrape(year_span=...)` |
| `scraper.py` | `BASE_URL`, `NEWBUILDINGS_QUERY`, `HEADERS` | Two-pass scrape; `infer_newbuild_from_text` only for legacy CSV in preprocessing |
| `scraper.py` | `random.uniform(1.0, 2.5)` | Inter-request delay |
| `preprocessing.py` | Percentiles, area/price cutoffs | Outlier policy |

---

## 9. Operational notes

**Rate limiting and ethics**

- Use polite delays; do not reduce sleep aggressively for bulk runs.
- Data is for academic analysis; respect Krisha.kz Terms of Use.

**Reproducibility**

- Scraped data changes daily; commit **methodology** and **code**; raw CSV is optional in version control.

**Python environment**

- On macOS, system Python may show `urllib3` / OpenSSL warnings; they are usually non-fatal.

---

## 10. Troubleshooting

| Symptom | Likely cause | Action |
|---------|--------------|--------|
| `ModuleNotFoundError` | Missing venv / packages | `pip3 install -r requirements.txt` |
| Few or zero rows after scrape | Blocked or HTML changed | Update selectors in `scraper.py`; verify `div.a-card` |
| `FileNotFoundError: krisha_raw.csv` | Clean/visualize before scrape | Run `scraper.py` or `main.py` |
| Spearman fails | Old pandas without scipy | Install `scipy` |
| Empty charts | Empty cleaned DataFrame | Check cleaning thresholds |

---

## 11. Cursor project rule

`.cursor/rules/run-python-check.mdc` reminds agents to run Python files after edits and lists expected artifacts per module.

---

*Last updated to match the codebase in this repository.*
