# R validation of Hexkit's statistics

These notebooks check the statistics Hexkit runs through **First Order Engine (FOE)** against
independent R implementations. Each notebook calls the FOE engine the way Hexkit does, computes
the same quantity in R through `rpy2`'s `%%R` cells, and compares the two in a pass/fail table.
The last cell raises if any comparison fails, so each notebook also works as a test.

Scope: the FOE engines that Hexkit pages and `utility/automation_engine.py` import directly.
See [docs/foe-enginization-status.md](../docs/foe-enginization-status.md) for that list.

## Notebooks

| Notebook | FOE engine | Hexkit surface | R reference | Checks |
|---|---|---|---|---|
| [01_frequentist_ztest](01_frequentist_ztest.ipynb) | `FrequentistEngine.run_synthesis`, `compute_interval_difference`, `apply_sidak`, `estimate_monetary_impact_per_variant` | experiment analysis, small-sample z-test fallback, automation | identity-link binomial `glm` (Wald SE = unpooled SE, `dispersion = φ`), `DescTools::BinomDiffCI`, `msm::deltamethod` | 168 |
| [02_boschloo_exact](02_boschloo_exact.ipynb) | `BoschlooEngine` | small sample analysis | `exact2x2::boschloo`, `DescTools::BinomDiffCI(method = "score")` | 60 |
| [03_bayesian_beta_binomial](03_bayesian_beta_binomial.ipynb) | `BayesianEngine.run_probability_analysis`, `run_monetary_projection`, `get_lift_prior` | sequential (revenue overlay), experiment analysis, automation | exact numerical integration of the Beta posteriors with the lift-prior weight | 21 |
| [04_continuous_metrics](04_continuous_metrics.ipynb) | `ContinuousMetricEngine.run_comparison_suite` (heuristic, NB, Gamma, hurdle), `fit_gamma`, `fit_negbin`, `winsorize_series`, `run_business_case` | automation, user-level analysis, pre-test | `fBasics::dagoTest`, `car::leveneTest`, `anova`, `oneway.test`, `kruskal.test`, `wilcox.test`, `MASS::glm.nb`, `MASS::fitdistr`, exact Gamma MLE, `msm::deltamethod` | 55 |
| [05_pretest_planning](05_pretest_planning.ipynb) | `PretestEngine` sample size, power, MDE tables (static and forecast), RPV moments | pre-test analysis, automation | `power.prop.test`, `TrialSize::TwoSampleMean.Equality`, `pwr::pwr.norm.test`, simulation | 97 |
| [06_sequential_msprt](06_sequential_msprt.ipynb) | `SequentialEngine` LLR, boundaries, power, `process_test_trajectory` | sequential analysis | numerical integration of the mixture likelihood ratio, `pwr::pwr.norm.test` | 131 |
| [07_interaction_glm](07_interaction_glm.ipynb) | `InteractionEngine.run_interaction_analysis` | interaction analysis | `glm(cbind(conv, visitors - conv) ~ t1 * t2, binomial)` | 93 |

All 625 checks pass on the committed run (R 4.6.2, rpy2 3.6.9, scipy 1.18.1, statsmodels 0.15.0,
FOE at the SHA pinned in `../requirements.txt`).

### Tolerances

Most checks agree to 1e-6 relative or better. The looser ones are deliberate:

- **Boschloo p-values** (abs 5e-4): scipy and exact2x2 each search for the nuisance-parameter
  maximum numerically, on different grids.
- **Bayesian quantities** (abs 3e-3 on probabilities, 2–3% relative on expectations): FOE uses
  Monte Carlo and R uses exact integration, so the gap is Monte Carlo error. With 1M draws the
  observed differences are about 0.1%.
- **RPV moments** (1%): checked against a simulated sample.
- **Two-sided MDE** (1e-4 relative): FOE's closed form leaves out the far rejection tail.

## What the validation turned up

None of these is a bug in FOE, but each one matters for reading or extending the results:

- **`run_comparison_suite` with `unit = per_transaction` (the config default) drops zero rows
  before any test**, including the Negative Binomial count test. For "items per order" that is
  correct. For a per-visitor count KPI with real zeros, use `unit = per_visitor`. With the
  default, the NB is fit to positive counts only and gives a different answer (notebook 04, §2).
- **`ContinuousMetricEngine.fit_gamma` is more precise than `MASS::fitdistr`.** It agrees with the
  exact Gamma MLE to about 1e-5, while `fitdistr` is off by about 0.1% in the shape parameter.
  That is enough to move Gamma LRT p-values noticeably, so notebook 04 uses the exact MLE as the
  reference.
- **Mann-Whitney exact vs asymptotic:** scipy's `method="auto"` switches to the exact test when
  either arm has ≤ 8 rows and there are no ties. R's `wilcox.test` uses a different rule, so the
  R side reproduces scipy's.
- **`pwr.norm.test` solved for the effect size** uses a loose internal tolerance (~1e-4). Notebook
  05 inverts its power function with a tight `uniroot` instead.

## Setup

1. **Install R** (≥ 4.3). The CRAN installer from <https://cran.r-project.org> ships binary
   packages and is the easiest route. With Homebrew R (`brew install r`) packages compile from
   source and also need `brew install openssl@3 nlopt pkgconf`.
2. **Install the R packages:**
   ```bash
   Rscript r_validation/install_r_packages.R
   ```
3. **Install the Python side.** This uses Hexkit's own requirements, so FOE, scipy and
   statsmodels match the app's versions:
   ```bash
   pip install -r r_validation/requirements.txt
   ```
   `rpy2` finds R through `R_HOME` or `R` on your `PATH`.

## Running

Interactively:

```bash
cd r_validation && jupyter lab
```

Headless, as a test run (fails on the first notebook with a mismatch):

```bash
cd r_validation
for nb in 0*.ipynb; do jupyter nbconvert --to notebook --execute --inplace "$nb" || break; done
```

Run from inside `r_validation/` so the notebooks can import `validation_helpers.py`.

## Not covered yet

- `foe.pretest.forecasting.TrafficForecastingEngine` (Prophet traffic forecast, automation
  only). This could be checked against R's `prophet` package, but fitted Prophet models are
  only loosely reproducible across languages.
- The CV-constant (unequal-variance) branch of the continuous sample-size formula has no
  off-the-shelf R equivalent.
- FOE modules Hexkit does not import yet (`foe.srm`, `foe.behavioral`, `foe.forecasting`,
  `foe.sequential.gsd_engine`). These can follow as they get wired into Hexkit.
- FOE helpers Hexkit does not call (non-inferiority, CUPED / Lin adjustment, aggregate φ
  estimation, bootstrap power).
