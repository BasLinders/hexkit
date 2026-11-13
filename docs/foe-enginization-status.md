# FOE Enginization Status

How much of Hexkit's statistics currently runs on **First Order Engine (FOE)** —
Bas Linders' standalone stats library (`first-order-engine`, imported as `foe`) —
versus how much still lives as bespoke logic inside Hexkit itself.

"Enginized" here means: the page/module imports and calls a `foe.*` engine as
its actual computation, not just that it *could*. Where a page has its own
parallel implementation of something FOE already provides, that counts as
**not enginized**, even if the two implementations happen to be equivalent.

Legend: ✅ enginized · ⚠️ partially enginized · ❌ not enginized ·
🏠 Hexkit-native by decision (no migration planned)

## Summary per page

| Page | Status | Runs on FOE | Still Hexkit-native |
|---|---|---|---|
| Pre-Test Analysis | ⚠️ Partial | `foe.pretest`, `foe.continuous` (sample size, MDE, continuous power, Gamma/NegBin fits) | Binomial power (statsmodels), seasonal Prophet forecast |
| SRM Check | ✅ Yes | `foe.srm` | — |
| Experiment Analysis | ⚠️ Partial | Bayesian path (`foe.bayesian`); frequentist difference CI and monetary impact (`foe.frequentist`) | Frequentist z-test, p-values, Šidák, power, non-inferiority, variance factor, inline SRM check |
| Small Sample Analysis | ✅ Yes | `foe.frequentist` (Boschloo, z-test synthesis, Šidák) | — |
| User Level Analysis | ⚠️ Partial | Business case only (`foe.continuous`) | Gamma/NegBin tests, outlier handling, ANOVA/Tukey/Kruskal/Mann-Whitney |
| Behavioral Analysis | ✅ Yes | `foe.behavioral` | Outlier *removal* (row filter; FOE has no removal step) |
| Interaction Analysis | ✅ Yes | `foe.interaction` | — |
| Sequential Analysis | ✅ Yes | `foe.sequential`, `foe.bayesian` (revenue overlay) | — |
| Automation (admin) | ✅ Yes | `foe.frequentist`, `foe.bayesian`, `foe.continuous`, `foe.pretest` incl. forecasting | — |
| Experimentation Growth (admin) | 🏠 Native | — | Sigmoid/Monte Carlo growth model (kept Hexkit-only by decision) |
| Home, Data Export | n/a | Data Export queries BigQuery via `foe.data` (see below) | No statistics |

## Summary per FOE module

| FOE module | Purpose | Used in Hexkit? |
|---|---|---|
| `foe.core` | Input/output models, validators, priors | ✅ Everywhere FOE is used |
| `foe.frequentist` | z-test, Šidák, power, non-inferiority, Boschloo, monetary impact | ⚠️ Fully in Small Sample + Automation; only partly in Experiment Analysis |
| `foe.bayesian` | Beta-Binomial Monte Carlo, lift prior, monetary projection | ✅ Experiment Analysis, Sequential, Automation, `utility/lift_prior_ui.py` |
| `foe.continuous` | Revenue/AOV metrics, Gamma/NegBin, comparison suite, business case | ⚠️ Pre-Test and Automation; User Level Analysis uses only the business case |
| `foe.pretest` | Sample size / MDE / power | ⚠️ Pre-Test (binomial power still via statsmodels), Automation |
| `foe.pretest.forecasting` | Prophet-based traffic forecasting | ⚠️ Automation only; Pre-Test seasonal mode still calls Prophet directly |
| `foe.sequential` | Always-valid LLR sequential testing | ✅ Sequential Analysis |
| `foe.srm` | Sample Ratio Mismatch (chi-squared) | ⚠️ SRM Check; Experiment Analysis still has its own inline chi-squared |
| `foe.behavioral` | Outliers, transforms, Welch's t-test | ✅ Behavioral Analysis |
| `foe.interaction` | Test-vs-test interaction (GLM) | ✅ Interaction Analysis |
| `foe.data` | BigQuery SQL building and querying | ✅ `utility/sql_builder.py`, `utility/bq_client.py` (with native fallback, see `USING_FOE_BACKEND`) |
| `foe.forecasting` | Conversion/revenue forecasting (holidays, covariates) | ❌ Not used anywhere |
| `foe.viz` | JSON-ready chart coordinates | ❌ Not used; every page builds its own charts |

---

## Detail per Hexkit surface

### ⚠️ `pages/pre_test_analysis.py` — Pre-test sample size, MDE & power
Mostly enginized. Sample size and MDE use `foe.pretest.operations.PretestEngine`,
continuous-KPI power uses `PretestEngine._power_two_sample`, and the Gamma /
NegBin baseline fits use `foe.continuous.operations.ContinuousMetricEngine`.

**Gaps:**
- **Seasonal ("Prophet Forecast") mode** calls `prophet.Prophet` directly in
  `run_prophet_forecast` instead of `foe.pretest.forecasting.TrafficForecastingEngine`.
  `utility/automation_engine.py` already uses the FOE engine for the same job,
  and its docstring notes the two are "functionally equivalent" but separately
  implemented.
- **Binomial power** (`calculate_power`) uses statsmodels' Cohen's h
  (`proportion_effectsize` + `zt_ind_solve_power`). FOE's
  `PretestEngine.calculate_power_binomial` uses an unpooled normal
  approximation instead, so switching changes the numbers slightly and needs a
  deliberate decision.
- **Private API use:** the page calls FOE's private helpers (`_relative_mde`,
  `_sample_size_per_variant`, `_power_two_sample`). These can change without
  notice when the pinned FOE commit is bumped.

### ✅ `pages/srm_calculator.py` — Sample Ratio Mismatch
Enginized: the chi-squared goodness-of-fit test runs in
`foe.srm.operations.SRMEngine.calculate_chi_squared` (α = 0.01). The page only
computes expected counts for the chart and table. Parity with the previous
native implementation was verified on 205 inputs.

### ⚠️ `pages/experiment_analysis.py` — Frequentist & Bayesian analysis
- **Bayesian path: enginized.** Uses `foe.bayesian.operations.BayesianEngine`
  (`run_probability_analysis`, `simulate_uplift_distributions`,
  `run_monetary_projection`) and `get_lift_prior`.
- **Frequentist path: partial.** Only the difference CI
  (`foe.frequentist.confidence.compute_interval_difference`) and the monetary
  impact (`FrequentistEngine.estimate_monetary_impact_per_variant` /
  `generate_monetary_conclusion`) come from FOE. `calculate_frequentist_statistics`
  still computes these in the page with `scipy.stats.norm`, although FOE has
  an equivalent for each:

  | Hexkit-native | FOE equivalent |
  |---|---|
  | z-statistics and p-values | `FrequentistEngine.run_ztest` / `run_synthesis` |
  | Šidák alpha | `foe.frequentist.operations.apply_sidak` |
  | Analytical and bootstrap power | `FrequentistEngine.calculate_analytical_power`, `run_vectorized_bootstrap_power` |
  | `calculate_non_inferiority` | `foe.frequentist.confidence.compute_non_inferiority` |
  | `calculate_timeseries_reduction_factor` | `FrequentistEngine.calculate_aggregate_variance_factor` |
  | Inline SRM check (`chisquare`, equal split) | `foe.srm.operations.SRMEngine.calculate_chi_squared` |

  The old version of this doc listed this page as fully enginized; that was
  wrong.

### ✅ `pages/small_sample_analysis.py` — Small-sample exact tests
Enginized: uses `foe.frequentist.boschloo.BoschlooEngine` for exact tests and
`FrequentistEngine.run_synthesis` / `apply_sidak` for the z-test comparison.

### ⚠️ `pages/user_level_analysis.py` — Revenue / per-user metrics
Mostly **not** enginized. Only the business case runs on FOE
(`ContinuousMetricEngine.run_business_case`). The page's own code does
everything else, although `ContinuousMetricEngine` has equivalents:

| Hexkit-native | FOE equivalent |
|---|---|
| `fit_gamma`, `fit_unit_model`, `perform_gamma_test`, `run_gamma_posthoc` | `fit_gamma`, `fit_unit_model`, `run_gamma_posthoc` |
| `is_count_kpi`, `fit_negbin`, `perform_negbin_test`, `run_negbin_posthoc` | `is_count_kpi`, `fit_negbin`, `run_negbin_posthoc` |
| `detect_outliers`, `winsorize_data` | `detect_outliers_ols`, `winsorize_series` |
| ANOVA / Tukey / Kruskal-Wallis / Mann-Whitney in `perform_stat_tests_and_conclusions` | `run_comparison_suite` |

Feature parity needs checking before migrating. For example, FOE's comparison
suite uses Welch's ANOVA (pingouin) for unequal variances and has no Tukey
post-hoc step. The old version of this doc listed this page as fully
enginized; that was wrong.

### ✅ `pages/behavioral_analysis.py` — Behavioral metric analysis
Enginized: outlier detection (IQR / OLS studentized residuals), winsorizing,
the log1p transform and Welch's t-test run in
`foe.behavioral.operations.BehavioralEngine`. The page keeps CSV preprocessing,
charts and outlier removal (a row filter; FOE has no removal step). Parity with
the previous native implementation was verified on 15 random datasets covering
both outlier methods.

### ✅ `pages/interaction_analysis.py` — Test-vs-test interaction analysis
Enginized: uses `foe.interaction.operations.InteractionEngine`
(`run_interaction_analysis`, `check_data_quality`). The previously inlined copy
of the engine has been removed. The page only computes the Wald CI bounds for
the forest plot.

### ✅ `pages/sequential_analysis.py` — Sequential (always-valid) testing
Enginized: the LLR boundaries and trajectory run in
`foe.sequential.operations.SequentialEngine` (`calculate_boundaries`,
`process_test_trajectory`). The Revenue Impact overlay uses
`foe.bayesian.operations.BayesianEngine.run_monetary_projection`.

### ✅ `utility/automation_engine.py` (powers `pages/automation.py`) — BigQuery → Airtable pipeline
The most fully enginized surface in Hexkit. Uses:
- `foe.frequentist.operations.FrequentistEngine`
- `foe.bayesian.operations.BayesianEngine`
- `foe.continuous.operations.ContinuousMetricEngine`
- `foe.pretest.operations.PretestEngine`
- `foe.pretest.forecasting.TrafficForecastingEngine` (seasonal traffic forecasting)

### ✅ `utility/sql_builder.py`, `utility/bq_client.py` — BigQuery data layer
Uses `foe.data` (`foe.data.sql.experiments`, `DataEngine`) when available and
falls back to Hexkit's own implementation otherwise. `USING_FOE_BACKEND` in
each module reports which path is active.

### 🏠 `pages/experimentation_growth.py` — Compound growth / win-rate projection
Hexkit-native **by decision**: this module stays in Hexkit and will not be
migrated to FOE. It implements its own sigmoid-based diminishing-returns model
and Monte Carlo simulation over a configurable win rate. No FOE module covers
this.

### ❌ `foe.viz` and `foe.forecasting` — not used anywhere
No page uses `foe.viz.operations.VizEngine`. Each page builds its own charts
(Altair, Plotly, Matplotlib/Seaborn). `foe.forecasting` (conversion/revenue
forecasting with holidays and covariates) is also unused.

### 🗄️ `hidden_pages/*` — legacy, pre-FOE
`bayesian_analysis.py`, `behavioral_analysis.py`, `frequentist_analysis.py`,
`interaction_non_scalable.py`, `mab_test.py` and
`experimentation_growth_deprecated.py` are older versions of the live pages.
They aren't in the app's navigation. Nothing to do beyond eventually deleting
them.

---

## Recommended order of remaining work

1. **Experiment Analysis frequentist core → `foe.frequentist` + `foe.srm`.**
   This is the biggest remaining gap, on one of the most-used pages. FOE
   already has equivalents for the z-test, Šidák, power, non-inferiority,
   variance factor and SRM check. Signatures differ (for example,
   `compute_non_inferiority` takes `se_diff` and `alpha`), and bootstrap power
   is random, so parity should be checked with tolerances.
2. **Pre-Test seasonal forecast → `foe.pretest.forecasting.TrafficForecastingEngine`.**
   This removes the second Prophet integration; `automation_engine.py` already
   uses the FOE version.
3. **User Level Analysis → `foe.continuous`.** A large migration. First decide
   on the differences in the comparison suite (Welch's ANOVA vs. standard ANOVA
   + Tukey).
4. **Pre-Test binomial power → `PretestEngine.calculate_power_binomial`.**
   Needs a decision first, because it changes the method from Cohen's h to an
   unpooled normal approximation. At the same time, replace the page's calls to
   FOE private helpers with public API.
5. **`foe.viz`.** Adopt when chart code gets consolidated; not urgent.

Done since the previous version of this doc (2026-09-14): Sequential Analysis,
Interaction Analysis, SRM Check and Behavioral Analysis. Small Sample Analysis
was built on FOE from the start. Experimentation Growth is now Hexkit-native by
decision.

*This document reflects the code as of 2026-11-13 (branch
`feature/srm-behavioral-engines`, FOE pinned at `ff19c01`). Re-check each
page's `foe` imports before relying on this for planning; both repos are under
active development.*
