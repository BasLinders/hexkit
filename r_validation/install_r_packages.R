# Installs the R packages the validation notebooks use as reference
# implementations. Run once:  Rscript r_validation/install_r_packages.R

pkgs <- c(
  "exact2x2",   # Boschloo's exact test                 (02_boschloo)
  "DescTools",  # Wald / Newcombe CIs for p1 - p2       (01, 02)
  "msm",        # delta method                          (01, 04)
  "cubature",   # 2-D integration for lift-prior check  (03_bayesian)
  "car",        # Levene test, type-II ANOVA            (04_continuous)
  "fBasics",    # D'Agostino K^2 omnibus normality test (04_continuous)
  "MASS",       # fitdistr (Gamma MLE), glm.nb          (04, 05)
  "TrialSize",  # two-sample z-test sample size         (05_pretest)
  "pwr"         # z-test power / MDE                    (05_pretest)
)

missing <- setdiff(pkgs, rownames(installed.packages()))
if (length(missing) > 0) {
  install.packages(missing, repos = "https://cloud.r-project.org")
}

for (p in pkgs) {
  suppressPackageStartupMessages(library(p, character.only = TRUE))
  cat(sprintf("%-10s %s\n", p, as.character(packageVersion(p))))
}
