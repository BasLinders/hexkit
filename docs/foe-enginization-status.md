# FOE Enginization Status

How much of Hexkit's statistics currently runs on **First Order Engine (FOE)** —
A standalone stats library (`first-order-engine`, imported as `foe`) —
versus how much still exists as bespoke logic inside Hexkit itself.

"Enginized" here means: the page/module imports and calls a `foe.*` engine as
its actual computation, not just that it *could*. Where a page has its own
parallel implementation of something FOE already provides, that counts as
**not enginized**, even if the two implementations happen to be equivalent.

Legend: ✅ enginized · ⚠️ partially enginized · ❌ not enginized ·
🏠 Hexkit-native by decision (no migration planned)

## Summary per page

| Page | Status | Runs on FOE | Still Hexkit-native |
|---|---|---|---|
| Pre-Test Analysis | ✅ Yes | `foe.pretest` incl. forecasting (MDE tables, sample size, power, seasonal forecast), `foe.continuous` (Gamma/NegBin fits) | — |
| SRM Check | ✅ Yes | `foe.srm` | — |
| Experiment Analysis | ✅ Yes | Bayesian path (`foe.bayesian`); frequentist z-test, p-values, significance, difference CI, Šidák, non-inferiority, variance factor φ, FPR/FNDR risk and monetary impact (`foe.frequentist`); SRM check (`foe.srm`) | Per-variant CIs (no FOE equivalent) |
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
| `foe.frequentist` | z-test, Šidák, power, non-inferiority, Boschloo, FPR/FNDR risk, variance factor, monetary impact | ✅ Small Sample, Automation, Experiment Analysis |
| `foe.bayesian` | Beta-Binomial Monte Carlo, lift prior, monetary projection | ✅ Experiment Analysis, Sequential, Automation, `utility/lift_prior_ui.py` |
| `foe.continuous` | Revenue/AOV metrics, Gamma/NegBin, comparison suite, business case | ⚠️ Pre-Test and Automation; User Level Analysis uses only the business case |
| `foe.pretest` | Sample size / MDE / power | ✅ Pre-Test, Automation |
| `foe.pretest.forecasting` | Prophet-based traffic forecasting | ✅ Pre-Test, Automation |
| `foe.sequential` | Always-valid LLR sequential testing | ✅ Sequential Analysis |
| `foe.srm` | Sample Ratio Mismatch (chi-squared) | ✅ SRM Check, Experiment Analysis |
| `foe.behavioral` | Outliers, transforms, Welch's t-test | ✅ Behavioral Analysis |
| `foe.interaction` | Test-vs-test interaction (GLM) | ✅ Interaction Analysis |
| `foe.data` | BigQuery SQL building and querying | ✅ `utility/sql_builder.py`, `utility/bq_client.py` (with native fallback, see `USING_FOE_BACKEND`) |
| `foe.forecasting` | Conversion/revenue forecasting (holidays, covariates) | ❌ Not used anywhere |
| `foe.viz` | JSON-ready chart coordinates | ❌ Not used; every page builds its own charts |

---

## Detail per Hexkit surface

### ✅ `pages/pre_test_analysis.py` — Pre-test sample size, MDE & power
Enginized, using public FOE API only:
- MDE tables: `PretestEngine.calculate_mde_table` (binomial) and
  `calculate_mde_table_continuous` (with `variance_scaling`), including the
  traffic-scenario sensitivity matrix
- sample size: `calculate_fixed_sample_size` /
  `calculate_fixed_sample_size_continuous`
- power: `calculate_power_binomial` / `calculate_power_continuous`
- seasonal mode: `foe.pretest.forecasting.TrafficForecastingEngine.run_seasonal_forecast`
  plus `PretestEngine.calculate_mde_from_forecast`, the same pair Automation
  uses
- Gamma / NegBin baseline fits and per-visitor moments:
  `ContinuousMetricEngine`, `PretestEngine.compound_per_visitor_moments`

The page no longer imports statsmodels or Prophet, and no longer calls FOE's
private helpers.

Since FOE `d6a9ec3` (#20, #21), power, sample size and MDE all target
`run_synthesis`'s unpooled z-test (variance p1q1 + p2q2, or σ² + σ²_treat).
Each MDE table is the exact inverse of the matching sample size, and power
is the inverse of that in turn, so the three modes agree. Checked against
the old page code on random inputs:
- Continuous with equal variance (3,000 inputs: MDE table at several
  traffic multipliers, sample size, power): identical.
- Binomial: sample size is 0.1–15% higher, power 0–0.3 points lower, and
  MDE 3–15% higher at realistic settings. MDE rises more only where it was
  already above 100%. The old formula used 2·p1q1 and under-powered:
  simulated real power fell to 74% for a planned 80% at p = 0.2% and a 50%
  MDE.
- Round trip: week n → MDE → sample size gives n back (±1 visitor), and
  power at that n equals the target within 3e-4. This holds for binomial and
  for continuous with CV-constant.
- Seasonal forecast: the Prophet point forecasts are identical to the old
  code. The interval bands vary from run to run (Prophet samples them), as
  they did before.

Behaviour changes:
- The continuous variance assumption (equal vs. constant CV) is now shown in
  MDE and seasonal mode too, not just sample size and power. Under constant
  CV a week with too little volume for any lift to reach the target power
  shows "N/A".
- Sample size mode now refuses a binomial MDE that would push the treatment
  rate above 100%. Before, it warned and computed with a clipped variance.
- Seasonal mode needs at least 14 days of history (FOE's minimum) and shows
  FOE's forecast summary under the chart.

### ✅ `pages/srm_calculator.py` — Sample Ratio Mismatch
Enginized: the chi-squared goodness-of-fit test runs in
`foe.srm.operations.SRMEngine.calculate_chi_squared` (α = 0.01). The page only
computes expected counts for the chart and table. Parity with the previous
native implementation was verified on 205 inputs.

### ✅ `pages/experiment_analysis.py` — Frequentist & Bayesian analysis
- **Bayesian path: enginized.** Uses `foe.bayesian.operations.BayesianEngine`
  (`run_probability_analysis`, `simulate_uplift_distributions`,
  `run_monetary_projection`) and `get_lift_prior`.
- **Frequentist path: enginized.** These come from FOE:
  - z-test, p-values, significance verdict and difference CI:
    `FrequentistEngine.run_synthesis` (unpooled SE scaled by φ, Šidák alpha
    at 3+ variants, CI that follows the selected tail)
  - Šidák alpha for display and FPR/FNDR: `foe.frequentist.operations.apply_sidak`
    (the same function `run_synthesis` uses)
  - variance factor φ from historical daily data:
    `FrequentistEngine.calculate_aggregate_variance_factor` (day-of-week
    controlled, n/(n−k) dof correction, clipped to [0.10, 5.0], regime)
  - non-inferiority: `foe.frequentist.confidence.compute_non_inferiority`
    (one-sided, at the page's confidence level, using the φ-adjusted
    unpooled SE)
  - SRM check: `foe.srm.operations.SRMEngine.calculate_chi_squared`
    (equal split, α = 0.01)
  - FPR / FNDR risk: `foe.frequentist.risk` (`false_positive_risk`,
    `fpr_at_p_value`, `false_negative_discovery_rate`, `resolve_prior`).
    These use the power the test was *planned* for, entered by the user,
    not post-hoc power.
  - monetary impact: `FrequentistEngine.estimate_monetary_impact_per_variant`
    / `generate_monetary_conclusion`

  Šidák, the SRM check, non-inferiority and the z-test were switched over on
  2026-11-17 (#117); the variance factor followed. Parity with the old inline code was checked on random inputs:
  - Šidák and SRM (5,000 inputs): Šidák matched exactly, the SRM p-value
    matched to within 2e-16, and there were no verdict flips.
  - Non-inferiority (20,000 inputs, including zero-SE cases): the p-value
    matched to within 1e-16, the lower bound matched exactly, and there were
    no verdict flips.
  - z-test via `run_synthesis` (20,000 inputs, all three tails, 2–5
    variants, φ from 0.3 to 3): the p-value matched to within 4e-16, the
    difference CI matched to within 6e-17, and there were no significance
    flips. Very small p-values are now more precise: FOE uses `norm.sf`,
    where the old `1 - norm.cdf` underflowed to 0.
  - Variance factor (5,000 synthetic histories, 14–120 days, campaign
    spikes, shuffled row order): φ, the clipped factor and the clipped
    flag matched exactly.

  Behaviour changes from the z-test switch:
  - Significance is now p < α (FOE's rule, the same as
    `foe_risk.assess_risk`). Before, it was p ≤ α.
  - When both arms are at 0% or 100% (unpooled SE = 0), FOE falls back to
    the pooled SE for the p-value and Newcombe's interval for the CI, and
    the page shows FOE's warning, which recommends an exact test.
    Identical arms now give p = 1 for every tail (one-sided used to give
    0.5). Opposite arms (0% vs 100%) are now significant in the tested
    direction; before, they were reported as flat. FPR/FNDR stay hidden
    for these cases.
  - A variant with 0 visitors now stops the analysis with "Every variant
    needs at least 1 visitor." FOE's input model rejects it.
  - `z_stats` was dropped from the results dict; nothing read it.

  Behaviour changes from the non-inferiority switch: FOE requires
  0 < margin < 1, so the page's margin input now starts at 0.1% instead of
  0%. With a zero SE, FOE reports p = 1.0 instead of NaN; the verdict is
  "not non-inferior" either way.

  Behaviour changes from the variance-factor switch: the page now uses FOE's
  regimes (stable < 0.80 ≤ neutral < 1.05 ≤ noisy < 1.50 ≤ high_noise)
  instead of its own (0.90 / 1.15 / 1.50). The visible change: φ between
  1.05 and 1.15 now gets the orange "SE inflated" message instead of the
  green "clean" one. The page keeps its own CSV validation
  (`validate_timeseries_data`) for readable error messages.

  Still computed in the page: the per-variant (marginal) conversion-rate
  CIs. FOE has no equivalent (`run_synthesis` only returns the difference
  CI), and they're simple enough to stay.

  The page no longer computes power at all, so there's nothing to migrate
  there. `norm.pdf` / `norm.ppf` in the distribution charts are chart code,
  not inference, and are out of scope until `foe.viz` is adopted.

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

1. **User Level Analysis → `foe.continuous`.** A large migration. First decide
   on the differences in the comparison suite (Welch's ANOVA vs. standard ANOVA
   + Tukey).
2. **`foe.viz`.** Adopt when chart code gets consolidated; not urgent.

Done since the previous version of this doc (2026-09-14): Sequential Analysis,
Interaction Analysis, SRM Check, Behavioral Analysis, Experiment Analysis
(frequentist path) and Pre-Test Analysis. Small Sample Analysis
was built on FOE from the start. Experimentation Growth is now Hexkit-native by
decision.

*This document reflects the code as of 2026-11-17 (branch `main`, FOE
pinned at `d6a9ec3`). Re-check each
page's `foe` imports before relying on this for planning; both repos are under
active development.*
