# FOE Enginization Status

How much of Hexkit's statistics currently runs on **First Order Engine (FOE)** —
Bas Linders' standalone stats library (`first-order-engine`, imported as `foe`) —
versus how much still lives as bespoke logic inside Hexkit itself.

"Enginized" here means: the page/module imports and calls a `foe.*` engine as
its actual computation, not just that it *could*. Where a page has its own
parallel implementation of something FOE already provides, that counts as
**not enginized**, even if the two implementations happen to be equivalent.

## Summary

| FOE module | Purpose | Enginized in Hexkit? |
|---|---|---|
| `foe.core` | Input/output models, validators, priors | ✅ Yes — used everywhere FOE is used |
| `foe.frequentist` | z-test, Šidák correction, bootstrap power, non-inferiority | ✅ Yes |
| `foe.bayesian` | Beta-Binomial Monte Carlo, lift prior, monetary projection | ✅ Yes |
| `foe.continuous` | Revenue/AOV metrics, CUPED, hurdle model | ✅ Yes |
| `foe.pretest` | Sample size / MDE tables | ✅ Yes |
| `foe.pretest` (forecasting) | Prophet-based traffic forecasting | ⚠️ Partial — only in the automation pipeline |
| `foe.sequential` | Always-valid LLR sequential testing | ❌ No — Hexkit has its own inline mSPRT/LLR code |
| `foe.srm` | Sample Ratio Mismatch (chi-squared) | ❌ No — Hexkit has its own chi-squared calculator |
| `foe.behavioral` | Funnel & segment analysis | ❌ No — Hexkit has its own statsmodels implementation |
| `foe.interaction` | Test-vs-test interaction/interference (GLM) | ❌ No — Hexkit has a fully duplicated `InteractionEngine` inlined |
| `foe.forecasting` | Conversion/revenue forecasting (holidays, covariates) | ❌ No — not used anywhere |
| `foe.viz` | JSON-ready chart coordinates | ❌ No — every page builds its own charts (Altair/Plotly/Matplotlib) |

---

## Detail per Hexkit surface

### ✅ `pages/experiment_analysis.py` — Frequentist & Bayesian analysis
Fully enginized for the frequentist path: uses `foe.frequentist.operations.FrequentistEngine`
and `foe.frequentist.confidence.compute_interval_difference`, plus `foe.core.models.AlternativeHypothesis`.

### ✅ `pages/user_level_analysis.py` — Revenue / per-user metrics
Fully enginized: uses `foe.continuous.operations.ContinuousMetricEngine` and
`foe.core.models.AnalysisUnit` / `AlternativeHypothesis`.

### ✅ `pages/pre_test_analysis.py` — Pre-test sample size & MDE
Mostly enginized: uses `foe.pretest.operations.PretestEngine` and
`foe.continuous.operations.ContinuousMetricEngine` for the core sample-size/MDE math.

**Gap:** the seasonal ("Prophet Forecast") mode calls `prophet.Prophet` directly
inline (`run_prophet_forecast`) instead of `foe.pretest.forecasting.TrafficForecastingEngine`.
This duplicates logic that `utility/automation_engine.py` already wraps around the
same FOE engine — the automation engine's own docstring calls out that this page
has "its own, independent Prophet-based forecast" that is "functionally
equivalent" but not shared. **Candidate for enginization.**

### ⚠️ `pages/sequential_analysis.py` — Sequential (always-valid) testing
**Not enginized for its core logic.** The mSPRT/LLR math (`calculate_msprt_llr`,
alpha/beta boundary logic) is implemented locally with `scipy.stats.norm`, not
via `foe.sequential.operations.SequentialEngine` — even though FOE ships a
dedicated sequential engine for exactly this ("Always Valid" LLR bounds).

FOE *is* used here, but only for an optional add-on: the ad-hoc Revenue Impact
overlay calls `foe.bayesian.operations.BayesianEngine` / `get_lift_prior`. The
primary sequential-testing feature this page exists for remains bespoke.
**High-value candidate for enginization** — this is the biggest gap between
what FOE already solves and what Hexkit still solves itself.

### ✅ `utility/automation_engine.py` (powers `pages/automation.py`) — BigQuery → Airtable pipeline
The most fully enginized surface in Hexkit — effectively Hexkit's own version
of the reference `first-order-pipeline` pattern described in FOE's README. Uses:
- `foe.frequentist.operations.FrequentistEngine`
- `foe.bayesian.operations.BayesianEngine`
- `foe.continuous.operations.ContinuousMetricEngine`
- `foe.pretest.operations.PretestEngine`
- `foe.pretest.forecasting.TrafficForecastingEngine` (seasonal traffic forecasting)

### ❌ `pages/srm_calculator.py` — Sample Ratio Mismatch
Not enginized. Implements its own chi-squared goodness-of-fit test with
`scipy.stats.chisquare`, duplicating `foe.srm.operations.SRMEngine`
one-for-one. Straightforward candidate for enginization — small surface, no
Prophet/statsmodels complexity to migrate.

### ❌ `pages/behavioral_analysis.py` — Behavioral / funnel metric analysis
Not enginized. Uses its own `statsmodels`-based preprocessing and modeling,
duplicating what `foe.behavioral.operations.BehavioralEngine` is meant to
provide (funnel & segment analysis). Candidate for enginization.

### ❌ `pages/interaction_analysis.py` — Test-vs-test interaction analysis
Not enginized, and notably: the file contains a **fully inlined copy** of an
`InteractionEngine` class (GLM/Binomial interaction model, coefficient regex
parsing, etc.) with a comment stating it's "inlined — no external import
required." This is a near-duplicate of `foe.interaction.operations.InteractionEngine`
and is the clearest case of logic that should be deleted from Hexkit and
replaced with the FOE import.

### ❌ `pages/experimentation_growth.py` — Compound growth / win-rate projection
Not enginized, and has no exact 1:1 FOE counterpart today. Implements its own
sigmoid-based diminishing-returns model and Monte Carlo simulation over a
configurable win rate. Conceptually adjacent to `foe.forecasting`, but that
FOE module is built for Prophet-based conversion/revenue forecasting with
holidays and covariates, not win-rate compounding — so this would need either
a new FOE capability or a deliberate decision to keep it Hexkit-side.

### ❌ `foe.viz` — not used anywhere
No page consumes `foe.viz.operations.VizEngine`. Every page builds its own
chart data/coordinates ad hoc (Altair in `sequential_analysis.py`/`srm_calculator.py`,
Plotly in `pre_test_analysis.py`, Matplotlib elsewhere). Enginizing this would
mostly pay off if/when chart logic needs to be consistent across pages or
reused in the automation pipeline's outputs.

### 🗄️ `hidden_pages/*` — legacy, pre-FOE
`bayesian_analysis.py`, `frequentist_analysis.py`, `interaction_non_scalable.py`,
`mab_test.py`, `experimentation_growth_deprecated.py` are earlier, hand-rolled
precursors to the now-live pages, already superseded and not part of the
active navigation. No action needed beyond eventual removal — they don't
represent open enginization work.

---

## Recommended order of work

1. **Sequential analysis → `foe.sequential`** — biggest gap; FOE's LLR engine
   already exists and directly replaces the bespoke `calculate_msprt_llr` logic.
2. **Interaction analysis → `foe.interaction`** — lowest risk, since the code
   is already a near-verbatim copy of the FOE engine; swapping the import
   removes duplicate maintenance immediately.
3. **SRM calculator → `foe.srm`** — small, self-contained, low risk.
4. **Pre-test seasonal forecast → `foe.pretest.forecasting.TrafficForecastingEngine`**
   — removes a second, independent Prophet integration that already has a
   known-equivalent FOE counterpart in use elsewhere (`automation_engine.py`).
5. **Behavioral analysis → `foe.behavioral`** — larger migration (custom
   preprocessing + statsmodels), needs a closer look at feature parity.
6. **Experimentation growth** — needs a product decision first: extend FOE
   with a growth-projection module, or keep this Hexkit-native.
7. **`foe.viz`** — adopt opportunistically once chart-building logic is
   consolidated, not urgent on its own.

*This document reflects the code as of 2026-09-14. Re-check `automation_engine.py`'s
module-level docstring and each page's imports before relying on this for
planning — both repos are under active development.*
