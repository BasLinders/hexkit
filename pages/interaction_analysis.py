import itertools
import textwrap
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from interaction_engine import InteractionEngine

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Interaction Analysis", page_icon="🔢")

# Singleton engine — constructed once per session
_engine = InteractionEngine()


# ---------------------------------------------------------------------------
# Sidebar / input
# ---------------------------------------------------------------------------

def _build_sidebar() -> Tuple[List[Dict[str, Any]], int]:
    """
    Renders the sidebar configuration widgets and returns the test configs
    and the number of tests.  Does NOT render the data editor.
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

    return test_configs, num_tests


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
    test_configs, _ = _build_sidebar()
    test_names = [cfg["name"] for cfg in test_configs]

    default_df = _build_default_df(test_configs, test_names)

    # --- Session-state persistence -------------------------------------------
    # Rebuild the default only when the set of columns changes (i.e. the user
    # reconfigured the tests), otherwise keep whatever the user typed in.
    current_cols = list(default_df.columns)
    if (
        "input_df" not in st.session_state
        or list(st.session_state["input_df"].columns) != current_cols
    ):
        st.session_state["input_df"] = default_df

    st.write("### Experiment Data Entry")
    st.markdown(
        f"Fill in data for all **{len(default_df)}** unique segments below. "
        "The **first variant** listed for each test is treated as the control."
    )

    edited_df = st.data_editor(
        st.session_state["input_df"],
        num_rows="fixed",
        use_container_width=True,
        hide_index=True,
    )
    # Persist edits immediately so they survive reruns
    st.session_state["input_df"] = edited_df

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

def render_model_summary(model) -> Dict[str, str]:
    """
    Renders the coefficient summary table and interaction analysis section.
    Returns the readable name mapping so downstream plots can reuse it.
    """
    summary_df = InteractionEngine.format_summary_table(model)
    # Build a name map from raw index → readable index for use in plots
    raw_index = model.summary2().tables[1].index
    name_map = dict(zip(raw_index, summary_df.index))

    st.write("### Model Summary")
    st.dataframe(summary_df.astype(float).round(4), use_container_width=True)

    # --- Interaction analysis -----------------------------------------------
    p_values = model.pvalues
    interaction_mask = p_values.index.str.contains(":")
    significant_interactions = p_values[(p_values < 0.05) & interaction_mask]

    st.write("### Interaction Analysis")
    if not significant_interactions.empty:
        st.warning(
            f"Detected **{len(significant_interactions)}** significant interaction(s):"
        )
        for raw_name, pval in significant_interactions.items():
            coef = model.params[raw_name]
            display_name = name_map.get(raw_name, raw_name)
            direction = "Positive (Synergy) 📈" if coef > 0 else "Negative (Clash) 📉"
            st.write(
                f"- **{display_name}** — p={pval:.2e}, "
                f"Coef: {coef:.4f} ({direction})"
            )
    else:
        st.success("No significant test interactions detected.")

    return name_map


def render_forest_plot(model, name_map: Dict[str, str]) -> None:
    """
    Forest plot of all coefficients (excluding intercept).

    Y-axis uses the human-readable labels from name_map.
    Long labels are wrapped automatically.
    """
    st.write("### Coefficient Forest Plot (Effect vs. Control Baseline)")
    st.info(
        "Dots show effect size (log-odds). Horizontal lines are 95% CIs. "
        "A line crossing 0 means the effect is not statistically significant."
    )

    params = model.params[1:]   # drop intercept
    conf   = model.conf_int()[1:]

    results = pd.DataFrame(
        {
            "Feature":     [name_map.get(n, n) for n in params.index],
            "Coefficient": params.values,
            "Lower":       conf[0].values,
            "Upper":       conf[1].values,
        }
    ).sort_values("Coefficient")

    # Wrap long labels so they don't overflow the axis
    wrapped_labels = [textwrap.fill(str(lbl), width=45) for lbl in results["Feature"]]

    fig, ax = plt.subplots(figsize=(10, max(len(results) * 0.6 + 2, 4)))

    for i, (_, row) in enumerate(results.iterrows()):
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
    ax.set_yticks(range(len(results)))
    ax.set_yticklabels(wrapped_labels, fontsize=9)
    ax.set_xlabel("Log-Odds Effect Size")
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()

    st.pyplot(fig)


def render_interaction_table(model, name_map: Dict[str, str]) -> None:
    """
    Displays a table of all interaction-term coefficients and their p-values.
    Replaces the placeholder heatmap with something actually useful.
    """
    st.write("### Interaction Term Details")

    params = model.params
    pvals  = model.pvalues
    conf   = model.conf_int()

    interaction_idx = [n for n in params.index if ":" in n]
    if not interaction_idx:
        st.info("No interaction terms found in the model.")
        return

    rows = []
    for raw in interaction_idx:
        rows.append(
            {
                "Interaction":   name_map.get(raw, raw),
                "Coefficient":   round(params[raw], 4),
                "p-value":       round(pvals[raw], 4),
                "CI Lower":      round(conf.loc[raw, 0], 4),
                "CI Upper":      round(conf.loc[raw, 1], 4),
                "Significant":   "✅" if pvals[raw] < 0.05 else "—",
                "Direction":     "Synergy 📈" if params[raw] > 0 else "Clash 📉",
            }
        )

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Static content sections
# ---------------------------------------------------------------------------

def render_info_expanders(test_names: List[str]) -> None:
    with st.expander("Why Use This Tool?"):
        st.markdown("""
### The Problem: Interaction Bias
When you run multiple experiments simultaneously, you risk **Interaction Bias** — the effect
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
1. **Define Tests & Variants** — Name your tests and list variants separated by commas (e.g. `A, B, C`).
2. **The Matrix** — The table auto-generates every possible variant combination.
3. **First is Baseline** — The **first variant** listed for each test is the statistical control.
4. **Fill & Calculate** — Enter visitor/conversion counts and click *Calculate*.
        """)

    with st.expander("Methodology & Statistical Approach"):
        st.markdown(r"""
This tool uses a **Generalized Linear Model (GLM)** with a Binomial family (logistic regression).

The response is modelled as a **two-column binomial** `(conversions, non_conversions)` — the
statistically correct approach for pre-aggregated count data. This avoids inflating the effective
sample size (which `freq_weights` would do), ensuring that standard errors and p-values are reliable.

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

        # --- Prepare data & fit model via engine ---
        try:
            df_prepared = InteractionEngine.prepare_aggregated_format(edited_df, test_cols)
            model = _engine.fit_interaction_model_from_formula(df_prepared, test_cols)
        except ValueError as exc:
            st.error(f"Model fitting failed: {exc}")
            st.stop()

        # --- Render results ---
        name_map = render_model_summary(model)
        render_forest_plot(model, name_map)
        render_interaction_table(model, name_map)


if __name__ == "__main__":
    run()
