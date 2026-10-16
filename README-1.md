# OOP with Python: Assignment 2 (Advent 2026)

MSCS & MSDS, **OOP with Python**. Five mini-projects in a Ugandan context, built as small, reusable, tested Python software. Each project has one notebook at the repository root, its reusable logic in `src/`, and its unit tests in `tests/`.

| # | Mini-project | Notebook | Reusable code in `src/` | Tests |
|---|---|---|---|---|
| 1 | UBOS District Population Forecaster | `project1_population.ipynb` | `population.py`, `forecasting.py`, `analysis.py` | 13 |
| 2 | Solar Micro-Grid Dispatch Planner | `project2_solar_microgrid.ipynb` | `microgrid.py` | 9 |
| 3 | Lake Victoria Fish Stock & Export Risk Model | `project3_fish_stock.ipynb` | `fishmodel.py` | 12 |
| 4 | Rainfall Pattern & Crop Suitability Analyser | `project4_rainfall.ipynb` | `rainfall/` (package) | 25 |
| 5 | Taxi Route Revenue, Pricing & Fleet Planner | `project5_taxi.ipynb` | `route.py`, `forecasters.py`, `planning.py`, `route_planner.py`, `plotting.py` | 11 |

---

## 1. Repository structure

```
.
├── README.md
├── requirements.txt
├── pytest.ini                         # tells pytest where src/ is
├── conftest.py                        # lets pytest find the src/ package
├── project1_population.ipynb
├── project2_solar_microgrid.ipynb
├── project3_fish_stock.ipynb
├── project4_rainfall.ipynb
├── project5_taxi.ipynb
├── src/                               # reusable logic, imported by the notebooks
│   ├── population.py                  # Project 1: DistrictPopulation
│   ├── forecasting.py                 # Project 1: Forecaster (abstract) + 3 models
│   ├── analysis.py                    # Project 1: growth, CAGR, error metrics, classrooms
│   ├── microgrid.py                   # Project 2: MicroGrid, HybridMicroGrid
│   ├── fishmodel.py                   # Project 3: FishStock, PriceModel, RiskAssessor
│   ├── rainfall/                      # Project 4 package
│   │   ├── constants.py
│   │   ├── region.py
│   │   ├── crop_rules.py
│   │   ├── similarity.py
│   │   ├── seasons.py
│   │   └── plots.py
│   ├── route.py                       # Project 5: Route
│   ├── forecasters.py                 # Project 5: Forecaster + subclasses
│   ├── planning.py                    # Project 5: equilibrium, backtesting, fleet sizing
│   ├── route_planner.py               # Project 5: RoutePlanner (composition)
│   └── plotting.py                    # Project 5: actual-vs-forecast plot
└── tests/                             # pytest unit tests for all five projects
```

`demand_30_days.csv` (Project 2) is created by its notebook the first time it runs. It is seeded, so it is identical every time.

## 2. How to set up

You need Python 3.10 or newer.

```bash
# 1. Clone the repository
git clone <your-repository-link>
cd <repository-folder>

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install the libraries
pip install -r requirements.txt
```

## 3. How to run

**The notebooks.** Start Jupyter **from the repository root** (the folder that contains `src/`), because the notebooks find the code through the `src/` folder. Open a notebook, then choose **Kernel > Restart & Run All**. Each notebook runs from top to bottom without errors.

```bash
jupyter notebook
```

**The tests.** Run them from the repository root.

```bash
pytest -v
```

Project 2 uses simulated typing for its `input()` demo, so *Restart & Run All* never stops. To try the real prompts, uncomment the line `# interactive_session(grid)` in Part 2 of that notebook.

## 4. Reproducibility

All randomness goes through `np.random.default_rng(seed)` with fixed seeds (42 in most places), so every run gives the same results.

- **Project 1:** the bootstrap intervals use seed 42.
- **Project 2:** `SEED = 42` is set in Part 0. The CSV and Monte Carlo results are identical on every run.
- **Project 3:** `PriceModel` uses default seed 42. The notebook uses seeds 42, 1 and 7 for the single path, the Monte Carlo and the 5-year runs.
- **Project 4:** the only random numbers are in the extension's synthetic demo data (seed 42).
- **Project 5:** `SEED = 42` is used through `np.random.default_rng`.

---

## Mini-Project 1: UBOS District Population Forecaster

Forecasts the 5-year population of five Ugandan districts (Kampala, Wakiso, Gulu, Mbarara and Jinja), validates three forecasting models, and estimates the extra primary-school classrooms each district needs by 2029.

### Libraries used

| Library | Used for |
|---|---|
| `numpy` | Arrays for years and populations, `np.polyfit`/`np.polyval` (linear trend), `np.var`/`np.std`, `np.percentile`, and `np.random.default_rng(42)` for the bootstrap |
| `statistics` (standard library) | Mean, median, sample variance and standard deviation, compared with NumPy's results (the `ddof` explanation) |
| `pandas` | Displaying results as tables (growth, model comparison, forecasts, classroom plan) |
| `matplotlib` | One figure of subplots (actual, fitted, forecast, train/test shading) and the shaded 95% intervals |
| `abc` (standard library) | The abstract base class `Forecaster` |
| `pytest` | Unit tests |

No black-box forecasting library is used. The three models are implemented by hand.

### OOP design

- **`DistrictPopulation`** validates its input (equal lengths, no empty, NaN or negative values) and uses the dunder methods `__len__` and `__repr__`.
- **Inheritance:** `Forecaster` is an abstract base class. `LinearTrendForecaster`, `ExponentialForecaster` and `FibonacciRatioForecaster` inherit from it and only write `fit` and `_curve`. `predict(horizon)` and `fitted()` are written once in the parent.
- Functions and classes have docstrings and type hints.

### Data

The population figures are **illustrative** (synthetic), as given in the brief. Mbarara and Jinja are invented for this project. The data can be replaced with UBOS figures in the `RAW_DATA` dictionary of the notebook.

### Summary of findings

- Wakiso grows fastest in relative terms (CAGR about 6.5% a year); Kampala and Gulu tie at about 4.6%.
- The exponential (CAGR) model has the lowest back-test error in all five districts. The Fibonacci-ratio model is far worse because it implies about 60% growth a year.
- About 4,420 additional classrooms are needed by 2029, over 80% of them in Wakiso and Kampala.
- See the *Findings & Limitations* section at the end of the notebook for the full interpretation.

---

## Mini-Project 2: Solar Micro-Grid Dispatch Planner

A rural health centre in Kasese draws energy from **solar panels (x)** and **batteries (y)**. Each day two demand constraints must be met:

```
3x + 2y = D1    (daytime load, kWh)
4x +  y = D2    (critical-equipment load, kWh)
```

This project models the micro-grid as a class, checks that the system is well-posed, solves 30 days of demand (loop vs vectorised), repairs physically impossible days, reports statistics and costs, and extends the model with a diesel generator and a Monte Carlo sensitivity test.

### Libraries used

| Library | Where it is used | How |
|---|---|---|
| `numpy` | everywhere | Arrays; `linalg.det`, `linalg.cond`, `linalg.matrix_rank` (well-posedness checks); `linalg.norm`, `linalg.lstsq`; `random.default_rng(42)` for every random number |
| `scipy.linalg.solve` | `MicroGrid.solve`, `solve_many` | Solves `A·v = d`; one call with a 2 x 30 right-hand side solves all days at once |
| `scipy.linalg.null_space` | Extension A | Finds the direction along which a dependent 3 x 3 system has infinitely many solutions |
| `scipy.optimize.nnls` | `MicroGrid.dispatch` | Non-negative least squares to repair infeasible days |
| `statistics` | Part 5 `describe()` | `mean`, `variance`, `stdev` of solar and battery use |
| `timeit` | Part 3 | Times the loop against the vectorised solve |
| `pandas` | CSV and tables | `date_range`, `read_csv` / `to_csv`, result tables |
| `matplotlib` | Parts 7 and Ext. B | Stacked bar chart with a second axis (`twinx`), histograms |
| `pathlib`, `tempfile` | CSV paths, tests | File paths; temporary files for tests |
| `pytest` | `tests/` | Unit tests |

### OOP design

- **`MicroGrid`** holds the coefficient matrix (`A`), labels, determinant, condition number and singularity flag. Methods: `report()`, `solve()`, `solve_day()`, `solve_many()`, `dispatch()`.
- **`HybridMicroGrid(MicroGrid)`** uses **inheritance**: it reuses the checks, `solve_many` and `dispatch`, calls `super().__init__`, and adds a diesel generator (3 x 3 system), a new `solve_day(d1, d2, d3)` and `solutions_when_dependent()`.
- Dunder methods: `__init__` only. Composition is not used in this project.
- Everything else is small, single-purpose functions with docstrings (`read_positive_number`, `generate_demand_csv`, `load_demand_csv`, `daily_cost`, `plot_dispatch`, `sensitivity_analysis`, `run_planner`).

### Testing

`pytest` runs 9 tests in `tests/test_microgrid.py`:

- **Normal cases:** known answer (D1 = 100, D2 = 85 gives 14 and 29 kWh), vectorised result equals day-by-day result, planted 3 x 3 answer, infeasible day repaired.
- **Edge cases:** zero demand, singular matrix refused, non-square matrix rejected, bad typed input (empty, text, `nan`, negative) rejected, CSV with a negative value refused.

### Data

All demand data is **synthetic (illustrative)**, generated by `generate_demand_csv()` with a weekday/weekend pattern, random noise and a few special days. Prices are illustrative.

### Summary of findings

- `det = -5`, `cond ≈ 5.8`: one unique solution, and forecast errors are amplified by at most about 6 times.
- Vectorised solving is about 40 times faster than a loop for 30 days and several hundred times faster at 100,000 days.
- 3 of 30 days were infeasible; NNLS left under half the unmet demand that clipping left.
- The battery is the more volatile source (std about 18.7 kWh, coefficient of variation about 0.67).
- Estimated monthly cost is about UGX 440,000 (illustrative prices: solar 150, battery 450 UGX/kWh).

The full **Findings & Limitations** section is the last cell of the notebook.

---

## Mini-Project 3: Lake Victoria Fish Stock & Export Risk Model

A fish-export cooperative in Jinja wants to know (a) whether its harvesting rate is sustainable and (b) how risky its revenue is. This project replaces an old Fibonacci stock model and a raw "variance > 50,000" risk flag with a logistic growth model, a bounded random-walk price model, and a CV and Monte Carlo risk assessment.

### Libraries used

| Library | What it is used for here |
|---|---|
| `numpy` | Arrays, seeded random numbers (`np.random.default_rng(seed)`), `clip`, `percentile` (VaR), mean/std/variance |
| `pandas` | Summary tables (scenario results, open vs closed season) |
| `matplotlib` | Stock trajectories, revenue histogram with VaR line, closed-season chart |
| `pytest` | Unit tests (12 tests) |
| `typing` (standard library) | Type hints in `src/fishmodel.py` |

No black-box forecasting libraries are used. The growth model, random walk, Monte Carlo and VaR are implemented by hand.

### Classes and how they work together

| Class / function | Role |
|---|---|
| `FishStock(r, K, N0, h)` | Logistic growth with harvesting: `N(t+1) = N + rN(1-N/K) - hN`. Provides `msy`, `equilibrium_stock`, `status`, and `simulate(weeks, closed_weeks)` (closed season for the extension). |
| `PriceModel(start, low, high, step_sd, seed)` | Weekly price as a random walk clipped to bounds. `simulate_paths(n_paths, weeks)` returns many paths at once. |
| `RiskAssessor(low_cv, high_cv, var_level)` | `classify(cv)` gives Low/Medium/High; `monte_carlo(fish, price_model, ...)` simulates annual revenue; `value_at_risk(...)` gives the 5% VaR. |
| `weekly_revenue`, `revenue_stats` | Revenue = tonnes x 1,000 x price; mean, median, variance, std, CV. |

`RiskAssessor.monte_carlo` receives a `FishStock` and a `PriceModel` object and uses them together. Every number (r, K, N0, bounds, thresholds, seed) is a parameter, so the model can be reused for another fishery.

### Data

No external dataset is used. All parameters come from the assignment brief or are stated assumptions in the notebook (for example the UGX 300 weekly price step and the closed-season weeks 20 to 27), so results are illustrative.

### Summary of findings

- h up to r/2 = 0.20 is sustainable and gives the highest revenue; the 52-week average catch is about 98% of MSY (1,000 t per week).
- h = 0.30 is overfishing: stock falls towards 2,500 t and the equilibrium catch (750 t per week) is below MSY.
- A raw variance threshold of 50,000 is meaningless (variance is in UGX squared and depends on the unit); the coefficient of variation has no units.
- Price risk is moderate: annual revenue CV is about 10% and the 5% VaR is roughly 15% below the mean.
- An 8-week closed season lowers 5-year revenue at h of 0.20 or less but raises it (about 7%) at h = 0.30.

The full 150 to 300 word interpretation is in the notebook under **Findings & Limitations**.

---

## Mini-Project 4: Rainfall Pattern & Crop Suitability Analyser

An agricultural extension officer wants to compare rainfall across regions of Uganda and advise farmers which months suit which crops. This project models regions and crop rules as classes, compares regions with three similarity/distance measures, detects rainy seasons with SciPy, and produces charts and a farmer advisory note.

### Modules used and how

| Module | Library used | What it does here |
|---|---|---|
| `region.py` | `numpy` | `Region` stores 12 monthly values in an array; sum, mean, argmax/argmin and std give the statistics. |
| `crop_rules.py` | `pandas` | `CropRule` holds a min/max monthly rainfall range and classifies each month as good, drought risk or waterlogging risk. |
| `similarity.py` | `numpy`, `pandas` | Cosine similarity (dot product ÷ product of norms), Pearson (`np.corrcoef`), Euclidean distance (`np.linalg.norm`) and a generic `pairwise_matrix()`. |
| `seasons.py` | `scipy.signal.find_peaks` | Finds rainy-season peaks. The year is repeated three times first so December and January are treated as neighbours. |
| `plots.py` | `matplotlib` | Rainfall line chart, crop-suitability heatmap and monthly box plots. |
| notebook | `scipy.spatial.distance` | Used only to *verify* our cosine similarity against SciPy. |
| notebook | `math` | Used only to *demonstrate* why `math.cos()` is the wrong tool for cosine similarity. |

### OOP design

- **Classes with attributes, methods, docstrings and type hints:** `Region` and `CropRule`.
- **Dunder methods:** `__init__` (builds and validates the object) and `__repr__` (readable printing) in both classes.
- **Association:** `CropRule.classify_region(region)` receives a `Region` and works on it. A `CropRule` does not store a region, so this is a "uses" relationship rather than true composition.
- **Class constants:** `CropRule.DROUGHT / GOOD / WATERLOGGING` replace unexplained numbers.
- **Inheritance:** not used, because nothing in this domain needed it.

### Data

- Monthly rainfall for Kampala, Gulu and Mbarara is **illustrative** (synthetic), taken from the assignment brief.
- Crop thresholds are converted from FAO seasonal and annual water requirements: Doorenbos & Kassam (1979), *Yield Response to Water*, FAO Irrigation and Drainage Paper 33 (maize, beans); FAO EcoCrop database (Robusta coffee).
- The box-plot extension currently uses synthetic data. Replace it with real monthly rainfall (NASA POWER or UNMA, 10+ years) for the extension.

### Tests

`pytest` runs 25 unit tests covering normal cases and edge cases (empty input, negative values, zero rainfall, threshold boundaries, constant rainfall, a December peak that needs year-wrapping).

### Summary of findings

- Kampala is the wettest and most evenly spread (1,600 mm, CV 37%); Gulu is the most seasonal (CV 61%); Mbarara is the driest (1,040 mm).
- Cosine similarity is high for every pair of regions (0.75 to 0.89) because rainfall is never negative; Pearson correlation is near zero. Only Euclidean distance shows differences in *amount*.
- Peak detection matched the known climate zone for Mbarara only. The illustrative data is too simple to reproduce Kampala's and Gulu's real patterns.
- In Mbarara, beans and coffee suit 8 months of the year and maize only 5.

The full *Findings & Limitations* section is at the end of the notebook.

---

## Mini-Project 5: Taxi Route Revenue, Pricing & Fleet Planner

Forecasts taxi-route passengers and revenue, finds the equilibrium fare from a supply-demand system, and sizes the fleet for each route (Ntinda, Entebbe and Mukono).

### Libraries used

| Library | Used for |
|---|---|
| `numpy` | Passenger arrays, vectorised revenue, `polyfit` for the trend line, `default_rng` for seeded simulation |
| `scipy` | `scipy.linalg.solve` solves the 2x2 supply-demand system |
| `statistics` | Mean, variance and standard deviation in `Route.stats()` |
| `matplotlib` | Actual vs forecast plots |
| `pandas` | Tidy result tables in the notebook only |
| `pytest` | Unit tests (11 tests: normal cases plus edge cases such as empty input, zero passengers and invalid use) |

No forecasting libraries are used. The forecasting methods are written by hand.

### OOP design

- **Classes with type hints and docstrings:** `Route`, `Forecaster`, `RoutePlanner`.
- **Inheritance:** `MovingAverage`, `ExponentialSmoothing`, `LinearTrend` and `SeasonalNaive` all extend `Forecaster` and override `forecast(history)`. Any forecaster can therefore be passed to the same `backtest()` function.
- **Composition:** `RoutePlanner` *has a* `Route` and *has a* `Forecaster`, and delegates to them.
- **Dunder methods:** `__repr__` on each class for readable printing.

### How to reuse it

- **New route:** add one line to `ROUTE_DATA` in the notebook.
- **New forecasting method:** subclass `Forecaster` and write `forecast(self, history)`.
- **Different fleet rules:** `plan_fleet(pax, trips_per_vehicle=10, seats=14, buffer=0.2)`.

### Assumptions

- Route data is **illustrative** (given in the assignment), not real data.
- The weekly-pattern data in the extension is **synthetic**: day 1 is a Monday, Friday is busiest, Sunday is quietest.
- A vehicle makes 8 one-way trips a day carrying 14 passengers, so it carries 112 passengers a day. A 15% buffer is added and the result is rounded **up**, with at least 1 vehicle per route.
- The backtest forecasts days 4 to 10, using only earlier days at each step.

### Summary of findings

- **Pricing:** the equilibrium Ntinda fare is **UGX 2,200** (76 passengers per trip-hour). The current UGX 2,000 fare is below it, so demand (80) exceeds supply (70).
- **Forecasting:** exponential smoothing (alpha = 0.9) had the lowest backtest error on all three routes.
- **Day-11 revenue forecast:** Ntinda about UGX 90,500, Entebbe about UGX 321,100, Mukono about UGX 150,400.
- **Fleet:** one vehicle per route is enough (capacity 112 passengers/day vs demand 45 to 65).
- **Extension:** on simulated weekly data, seasonal naive cut error by about 81% versus the 3-day moving average.

The full interpretation is in the **Findings & Limitations** section at the end of the notebook.

---

## AI use disclosure

I used an AI assistant (Claude, by Anthropic) as a coding, study and drafting helper across these projects, not as the author. It was used to:

- draft the first version of the notebook structures, the code in `src/`, the explanatory text and the pytest tests;
- suggest design ideas, such as the abstract-class design for the forecasters (Projects 1 and 5), using `scipy.optimize.nnls` instead of only clipping negative values and an energy-balance equation as the third constraint for the diesel extension (Project 2), verifying my cosine similarity against `scipy.spatial.distance.cosine` and handling the circular year before peak detection (Project 4), and separating logic into `src/`, adding type hints and walk-forward backtesting (Project 5);
- **help debug** the code in all five projects;
- help check results and review the code;
- **design and structure this README**, including its headings and layout;
- **design the headings and other markdown content in the notebooks.**

I reviewed all AI-suggested code, ran every notebook with *Restart & Run All* and the full test suite, checked results by hand where possible (for example, in Project 2, D1 = 100 and D2 = 85 give solar 14 kWh and battery 29 kWh), and I take responsibility for the submitted work. I can explain every part of the code.
