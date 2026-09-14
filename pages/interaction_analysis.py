import itertools
import textwrap
from typing import Any, Dict, List, Tuple
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from scipy.stats import norm
from foe.interaction.operations import InteractionEngine

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Interaction Analysis", page_icon="🔢", layout="wide")

# Model fitting, data-quality checks, and coefficient naming/conclusions are
# delegated entirely to the shared FOE InteractionEngine
# (foe.interaction.operations) rather than duplicated here.
_engine = InteractionEngine()

# Confidence intervals below are derived from FOE's (coefficient, std_err)
# output as coef ± Z * std_err, which is numerically identical to
# statsmodels' own GLM conf_int() at the matching alpha (both are Wald/z
# based since the Binomial family fixes the dispersion at 1).
ALPHA = 0.05
_Z_CRIT = float(norm.ppf(1 - ALPHA / 2))


# ---------------------------------------------------------------------------
# Sidebar / input
# ---------------------------------------------------------------------------

def _build_sidebar() -> List[Dict[str, Any]]:
    """
    Renders the sidebar configuration widgets and returns the test configs.
    """
    st.sidebar.header("Configuration")

    # Keep the sidebar cap in sync with the engine constant
    num_tests = st.sidebar.number_input(
        "Number of concurrent tests",
        min_value=1,
        max_value=InteractionEngine.MAX_TESTS,  # single source of truth
        value=2,
    )

    test_configs: List[Dict[str, Any]] = []
    for i in range(num_tests):
        st.sidebar.markdown("---")
        name = st.sidebar.text_input(
            f"Test {i + 1} Name", value=f"test_{i + 1}", key=f"t_name_{i}"
        )
        variants_str = st.sidebar.text_input(
            f"Variants for {name} (comma-separated)",
            value="A, B",
            key=f"t_vars_{i}",
        )
        variants = [v.strip() for v in variants_str.split(",") if v.strip()]
        test_configs.append({"name": name, "variants": variants})

    return test_configs


def _build_default_df(
    test_configs: List[Dict[str, Any]], test_names: List[str]
) -> pd.DataFrame:
    """Returns a zero-filled dataframe covering every variant combination."""
    all_variant_levels = [cfg["variants"] for cfg in test_configs]
    combinations = list(itertools.product(*all_variant_levels))

    init_data: List[Dict[str, Any]] = []
    for combo in combinations:
        row: Dict[str, Any] = {test_names[i]: str(combo[i]) for i in range(len(test_names))}
        row["visitors"] = 0
        row["conversions"] = 0
        init_data.append(row)

    df = pd.DataFrame(init_data)
    df["visitors"] = df["visitors"].astype(int)
    df["conversions"] = df["conversions"].astype(int)
    return df


def _df_signature(df: pd.DataFrame, test_names: List[str]) -> str:
    """
    Returns a string that changes whenever the table schema OR the set of
    variant combinations changes.  Columns alone are insufficient because
    adding a variant (e.g. "A, B" → "A, B, C") doesn't change column names.
    """
    col_sig = str(list(df.columns))
    row_sig = str([tuple(r) for r in df[test_names].itertuples(index=False)])
    return col_sig + row_sig


def user_input() -> Tuple[pd.DataFrame, List[str]]:
    """
    Full input section: sidebar + data editor.

    Uses st.session_state to persist the edited table across Streamlit reruns
    so that users don't lose data when they click Calculate or change sidebar
    options.

    Returns
    -------
    edited_df : pd.DataFrame
    test_names : list[str]
    """
    test_configs = _build_sidebar()
    test_names = [cfg["name"] for cfg in test_configs]

    default_df = _build_default_df(test_configs, test_names)

    # --- Session-state persistence -------------------------------------------
    # Include variant-row structure in the signature, not just column names,
    # so that adding/removing variants correctly triggers a table rebuild.
    sig = _df_signature(default_df, test_names)
    schema_changed = st.session_state.get("_col_sig") != sig

    if schema_changed:
        st.session_state["_col_sig"] = sig
        st.session_state["input_df"] = default_df
        st.session_state.pop("_data_editor", None)

    st.write("### Experiment Data Entry")
    st.markdown(
        f"Fill in data for all **{len(default_df)}** unique segments below. "
        "The **first variant** listed for each test is treated as the control."
    )

    edited_df = st.data_editor(
        st.session_state["input_df"],
        key="_data_editor",
        num_rows="fixed",
        width="stretch",
        hide_index=True,
    )
    
    # DO NOT write edited_df back to st.session_state["input_df"] here.
    # Just return it for downstream calculations.

    return edited_df, test_names


# ---------------------------------------------------------------------------
# Validation (UI layer — user-friendly messages)
# ---------------------------------------------------------------------------

def validate_input_df(df: pd.DataFrame, test_cols: List[str]) -> List[str]:
    """
    Returns a list of human-readable error strings.
    An empty list means the data is ready to model.
    """
    errors: List[str] = []

    if df.empty:
        errors.append("The data table is empty.")
        return errors

    if df["visitors"].le(0).any():
        errors.append("Every row must have at least 1 visitor.")

    if df["conversions"].lt(0).any():
        errors.append("'conversions' cannot be negative.")

    if (df["conversions"] > df["visitors"]).any():
        bad_rows = df[df["conversions"] > df["visitors"]][test_cols].to_string(index=False)
        errors.append(
            f"Some rows have more conversions than visitors:\n```\n{bad_rows}\n```"
        )

    if df[test_cols].isnull().any().any():
        errors.append("Variant columns contain empty cells.")

    return errors


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def render_model_summary(results: List[Dict[str, Any]]) -> None:
    """
    Renders the coefficient summary table and interaction analysis section.

    `results` is the list of per-term dicts returned by
    InteractionEngine.run_interaction_analysis (term, raw_term, coefficient,
    std_err, z_score, p_value, is_significant, conclusion).
    """
    summary_df = pd.DataFrame(
        [
            {
                "Coef.": r["coefficient"],
                "Std.Err.": r["std_err"],
                "z": r["z_score"],
                "P>|z|": r["p_value"],
            }
            for r in results
        ],
        index=[r["term"] for r in results],
    )

    st.write("### Model Summary")
    st.dataframe(summary_df.astype(float).round(4), width="stretch")

    # --- Interaction analysis -----------------------------------------------
    significant_interactions = [
        r for r in results if ":" in r["raw_term"] and r["p_value"] < ALPHA
    ]

    st.write("### Interaction Analysis")
    if significant_interactions:
        st.warning(
            f"Detected **{len(significant_interactions)}** significant interaction(s):"
        )
        for r in significant_interactions:
            direction = "Positive (Synergy) 📈" if r["coefficient"] > 0 else "Negative (Clash) 📉"
            st.write(
                f"- **{r['term']}** — p={r['p_value']:.2e}, "
                f"Coef: {r['coefficient']:.4f} ({direction})"
            )
    else:
        st.success("No significant test interactions detected.")


def render_forest_plot(results: List[Dict[str, Any]]) -> None:
    """
    Forest plot of all coefficients (excluding the intercept/baseline term).

    Confidence intervals are derived from (coefficient, std_err) at ALPHA —
    see the module-level note on _Z_CRIT. Long labels are wrapped automatically.
    """
    st.write("### Coefficient Forest Plot (Effect vs. Control Baseline)")
    st.info(
        "Dots show effect size (log-odds). Horizontal lines are 95% CIs. "
        "A line crossing 0 means the effect is not statistically significant."
    )

    terms = [r for r in results if r["term"] != "Baseline (Control Group)"]

    plot_df = pd.DataFrame(
        {
            "Feature": [r["term"] for r in terms],
            "Coefficient": [r["coefficient"] for r in terms],
            "Lower": [r["coefficient"] - _Z_CRIT * r["std_err"] for r in terms],
            "Upper": [r["coefficient"] + _Z_CRIT * r["std_err"] for r in terms],
        }
    ).sort_values("Coefficient")

    # Wrap long labels so they don't overflow the axis
    wrapped_labels = [textwrap.fill(str(lbl), width=45) for lbl in plot_df["Feature"]]

    fig, ax = plt.subplots(figsize=(10, max(len(plot_df) * 0.6 + 2, 4)))

    for i, (_, row) in enumerate(plot_df.iterrows()):
        is_significant = not (row["Lower"] <= 0 <= row["Upper"])
        color = "#ff4b4b" if is_significant else "#7d7d7d"
        ax.errorbar(
            row["Coefficient"],
            i,
            xerr=[[row["Coefficient"] - row["Lower"]], [row["Upper"] - row["Coefficient"]]],
            fmt="o",
            color=color,
            capsize=3,
            markersize=8,
        )

    ax.axvline(0, color="black", linestyle="--", alpha=0.5)
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(wrapped_labels, fontsize=9)
    ax.set_xlabel("Log-Odds Effect Size")
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)


def render_interaction_table(results: List[Dict[str, Any]]) -> None:
    """
    Displays a table of all interaction-term coefficients and their p-values.
    """
    st.write("### Interaction Term Details")

    interaction_terms = [r for r in results if ":" in r["raw_term"]]
    if not interaction_terms:
        st.info("No interaction terms found in the model.")
        return

    rows = []
    for r in interaction_terms:
        lower = r["coefficient"] - _Z_CRIT * r["std_err"]
        upper = r["coefficient"] + _Z_CRIT * r["std_err"]
        rows.append(
            {
                "Interaction": r["term"],
                "Coefficient": round(r["coefficient"], 4),
                "p-value": round(r["p_value"], 4),
                "CI Lower": round(lower, 4),
                "CI Upper": round(upper, 4),
                "Significant": "✅" if r["p_value"] < ALPHA else "—",
                "Direction": "Synergy 📈" if r["coefficient"] > 0 else "Clash 📉",
            }
        )

    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# Static content sections
# ---------------------------------------------------------------------------

def render_info_expanders(test_names: List[str]) -> None:
    with st.expander("Why Use This Tool?"):
        st.markdown("""
            ### The Problem: Interaction Bias
            When you run multiple experiments simultaneously, you risk **Interaction Bias**; the effect
            of one change (e.g. a new button colour) being influenced by another change (e.g. new pricing).
            Standard A/B dashboards assume independence and can declare a "winner" that actually performs
            poorly when combined with other live features.

            ### Solution: Factorial Interaction Analysis
            This tool calculates the **Combined Effect** across every variant combination.

            - **Detect Clashes:** Two great features that hurt conversion when shown together.
            - **Discover Synergies:** Combinations where 1 + 1 = 3.
            - **Clean Results:** Remove noise caused by concurrent tests to isolate true lift.
        """)
        st.info(
            "Use this tool whenever you have overlapping traffic between two or more "
            "experiments to ensure your winning variants are truly compatible."
        )

    with st.expander("How to Use This Tool"):
        st.markdown("""
            1. **Define Tests & Variants** - Name your tests and list variants separated by commas (e.g. `A, B, C`).
            2. **The Matrix** - The table auto-generates every possible variant combination.
            3. **First is Baseline** - The **first variant** listed for each test is the statistical control.
            4. **Fill & Calculate** - Enter visitor/conversion counts and click *Calculate*.
        """)

    with st.expander("Methodology & Statistical Approach"):
        st.markdown(r"""
            This tool uses a **Generalized Linear Model (GLM)** with a Binomial family (logistic regression).

            The response is modelled as a **two-column binomial** `(conversions, non_conversions)`. This avoids inflating the effective
            sample size, ensuring that standard errors and p-values are reliable.

            **The Interaction Formula** (two tests):
            $$\text{logit}(p) = \beta_0 + \beta_1\,\text{Test}_1 + \beta_2\,\text{Test}_2 + \beta_3(\text{Test}_1 \times \text{Test}_2)$$

            The **interaction term** ($\beta_3$) tells us whether the combined effect of two variants differs
            significantly from the sum of their individual effects.
        """)

    with st.expander("SQL Query Helper (Get your data)"):
        st.markdown(
            "Group users by every experiment they were exposed to. "
            "Use the template below in your data warehouse:"
        )
        col_list   = ",\n    ".join([f"{n}_variant" for n in test_names])
        group_cols = ",\n    ".join([f"{n}_variant" for n in test_names])
        sql = (
            f"SELECT\n"
            f"    {col_list},\n"
            f"    COUNT(user_id)        AS visitors,\n"
            f"    SUM(conversion_flag)  AS conversions\n"
            f"FROM user_experiment_log\n"
            f"GROUP BY\n"
            f"    {group_cols}"
        )
        st.code(sql, language="sql")
        st.info(
            "Variant names in your query (e.g. `'A'`, `'B'`) must match what you "
            "type into the configuration panel exactly."
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> None:
    st.title("Interaction Analysis")
    st.markdown(
        "Detect synergies and clashes between concurrent A/B tests "
        "using a full-factorial logistic regression model."
    )

    edited_df, test_cols = user_input()

    render_info_expanders(test_cols)

    if st.button("Calculate Interaction Effects", type="primary"):
        # --- UI-level validation ---
        errors = validate_input_df(edited_df, test_cols)
        if errors:
            for err in errors:
                st.error(err)
            st.stop()

        # --- Fit model & analyze via the shared FOE InteractionEngine ---
        try:
            # Soft data-quality warnings shown before fitting so the user
            # understands the root cause of any perfect-separation messages.
            for msg in InteractionEngine.check_data_quality(edited_df, test_cols):
                st.warning(msg)

            results, fit_warnings = _engine.run_interaction_analysis(
                edited_df, test_cols, alpha=ALPHA
            )

            for msg in fit_warnings:
                st.warning(msg)
        except ValueError as exc:
            st.error(f"Model fitting failed: {exc}")
            st.stop()

        # --- Render results ---
        render_model_summary(results)
        render_forest_plot(results)
        render_interaction_table(results)


if __name__ == "__main__":
    run()
