"""
sql_builder.py
Generates BigQuery SQL for each export mode from structured user inputs.
No hardcoded limits on variants or experiments.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _table_ref(project: str, dataset: str) -> str:
    return f"`{project}.{dataset}.events_*`"


def _suffix_filter(start: str, end: str, alias: str = "") -> str:
    col = f"{alias}._TABLE_SUFFIX" if alias else "_TABLE_SUFFIX"
    return (
        f"{col} BETWEEN FORMAT_DATE('%Y%m%d', PARSE_DATE('%Y-%m-%d', '{start}'))\n"
        f"    AND FORMAT_DATE('%Y%m%d', PARSE_DATE('%Y-%m-%d', '{end}'))"
    )


# ---------------------------------------------------------------------------
# Dataclasses — one per mode
# ---------------------------------------------------------------------------

@dataclass
class VariantPair:
    label: str        # 'A', 'B', 'C', …
    string: str       # the actual variant ID string


@dataclass
class ExperimentConfig:
    experiment_id: str
    prefix: str           # for LIKE matching
    variants: list[VariantPair] = field(default_factory=list)


@dataclass
class BaselineParams:
    project: str
    dataset: str
    start_date: str
    end_date: str
    output_type: Literal["binomial", "revenue"] = "binomial"
    # "per_user" (one row per order, for pre-test model fitting) is only
    # meaningful for output_type == "revenue".
    output_shape: Literal["aggregate", "daily", "per_user"] = "aggregate"
    # Adds add-to-cart conversion counts alongside the purchase-based ones.
    # Binomial only — revenue mode has no "conversion" concept to extend.
    kpi_add_to_cart: bool = False
    page_filter_type: Optional[Literal["regex", "contains"]] = None  # None = no filter
    page_filter_value: str = ""


@dataclass
class BinomialParams:
    project: str
    dataset: str
    start_date: str
    end_date: str
    param_key: str                          # e.g. 'exp_variant_string'
    match_strategy: Literal["exact", "like"]
    experiments: list[ExperimentConfig]    # ≥1; multi-experiment = len > 1
    post_exposure_filter: bool = True
    # KPI toggles — zero-cost group
    kpi_transactions: bool = True
    kpi_add_to_cart: bool = True
    kpi_aov: bool = True
    kpi_ideal: bool = False
    kpi_device_split: bool = True
    # KPI toggles — cost-warning group (adds page_view scan)
    kpi_login: bool = False
    kpi_create_account: bool = False


@dataclass
class ContinuousParams:
    project: str
    dataset: str
    start_date: str
    end_date: str
    param_key: str
    match_strategy: Literal["exact", "like"]
    experiments: list[ExperimentConfig]
    device_filter: Literal["all", "desktop", "mobile"] = "all"
    query_mode: Literal["all_users", "revenue_only"] = "all_users"
    post_exposure_filter: bool = True


@dataclass
class SequentialParams:
    project: str
    dataset: str
    start_date: str
    end_date: str
    param_key: str
    experiments: list[ExperimentConfig]
    use_persistence: bool = True
    reset_cumulative_data: bool = False
    cumulative_table: str = ""            # project.dataset.table_name
    kpi_transactions: bool = True
    kpi_add_to_cart: bool = True
    kpi_aov: bool = True
    kpi_ideal: bool = False
    kpi_device_split: bool = True
    kpi_login: bool = False
    kpi_create_account: bool = False


@dataclass
class InteractionParams:
    project: str
    dataset: str
    start_date: str
    end_date: str
    param_key: str
    experiments: list[ExperimentConfig]   # each has exactly 2 variants (A, B)
    kpi_transactions: bool = True
    kpi_add_to_cart: bool = True


# ---------------------------------------------------------------------------
# BASELINE
# ---------------------------------------------------------------------------

def _baseline_page_filter(p: BaselineParams, table: str, suffix: str) -> tuple[str, str]:
    """Builds the optional page-view filter CTE + join clause shared by all baseline shapes."""
    if not (p.page_filter_type and p.page_filter_value):
        return "", ""
    if p.page_filter_type == "regex":
        page_condition = f"AND REGEXP_CONTAINS(params.value.string_value, r'{p.page_filter_value}')"
    else:
        page_condition = f"AND params.value.string_value LIKE '%{p.page_filter_value}%'"
    cte = f"""
view_page_users AS (
  SELECT DISTINCT user_pseudo_id
  FROM {table}, UNNEST(event_params) AS params
  WHERE {suffix}
    AND event_name = 'page_view'
    AND params.key = 'page_location'
    {page_condition}
),"""
    join = "INNER JOIN view_page_users ON main.user_pseudo_id = view_page_users.user_pseudo_id"
    return cte, join


def build_baseline(p: BaselineParams, limit: int = 0) -> str:
    if p.output_shape == "daily":
        return _build_baseline_daily(p, limit)
    if p.output_shape == "per_user":
        return _build_baseline_per_user(p, limit)
    return _build_baseline_aggregate(p, limit)


def _build_baseline_aggregate(p: BaselineParams, limit: int = 0) -> str:
    table = _table_ref(p.project, p.dataset)
    suffix = _suffix_filter(p.start_date, p.end_date)
    view_page_cte, join_clause = _baseline_page_filter(p, table, suffix)
    limit_clause = f"\nLIMIT {limit}" if limit else ""

    if p.output_type == "revenue":
        select_cols = """
  -- All devices
  COUNT(DISTINCT main.user_pseudo_id) AS total_visitors,
  COUNT(DISTINCT CASE WHEN main.event_name = 'purchase' THEN main.ecommerce.transaction_id END) AS total_transactions,
  SUM(CASE WHEN main.event_name = 'purchase' THEN main.ecommerce.purchase_revenue ELSE 0 END) AS total_purchase_revenue,
  ROUND(SUM(CASE WHEN main.event_name = 'purchase' THEN main.ecommerce.purchase_revenue ELSE 0 END) / NULLIF(COUNT(DISTINCT main.user_pseudo_id), 0), 2) AS total_revenue_per_visitor,
  ROUND(SUM(CASE WHEN main.event_name = 'purchase' THEN main.ecommerce.purchase_revenue ELSE 0 END) / NULLIF(COUNT(DISTINCT CASE WHEN main.event_name = 'purchase' THEN main.ecommerce.transaction_id END), 0), 2) AS total_average_order_value,
  -- Mobile
  COUNT(DISTINCT CASE WHEN main.device.category = 'mobile' THEN main.user_pseudo_id END) AS mobile_visitors,
  COUNT(DISTINCT CASE WHEN main.event_name = 'purchase' AND main.device.category = 'mobile' THEN main.ecommerce.transaction_id END) AS mobile_transactions,
  SUM(CASE WHEN main.event_name = 'purchase' AND main.device.category = 'mobile' THEN main.ecommerce.purchase_revenue ELSE 0 END) AS mobile_purchase_revenue,
  ROUND(SUM(CASE WHEN main.event_name = 'purchase' AND main.device.category = 'mobile' THEN main.ecommerce.purchase_revenue ELSE 0 END) / NULLIF(COUNT(DISTINCT CASE WHEN main.device.category = 'mobile' THEN main.user_pseudo_id END), 0), 2) AS mobile_revenue_per_visitor,
  ROUND(SUM(CASE WHEN main.event_name = 'purchase' AND main.device.category = 'mobile' THEN main.ecommerce.purchase_revenue ELSE 0 END) / NULLIF(COUNT(DISTINCT CASE WHEN main.event_name = 'purchase' AND main.device.category = 'mobile' THEN main.ecommerce.transaction_id END), 0), 2) AS mobile_average_order_value,
  -- Desktop
  COUNT(DISTINCT CASE WHEN main.device.category = 'desktop' THEN main.user_pseudo_id END) AS desktop_visitors,
  COUNT(DISTINCT CASE WHEN main.event_name = 'purchase' AND main.device.category = 'desktop' THEN main.ecommerce.transaction_id END) AS desktop_transactions,
  SUM(CASE WHEN main.event_name = 'purchase' AND main.device.category = 'desktop' THEN main.ecommerce.purchase_revenue ELSE 0 END) AS desktop_purchase_revenue,
  ROUND(SUM(CASE WHEN main.event_name = 'purchase' AND main.device.category = 'desktop' THEN main.ecommerce.purchase_revenue ELSE 0 END) / NULLIF(COUNT(DISTINCT CASE WHEN main.device.category = 'desktop' THEN main.user_pseudo_id END), 0), 2) AS desktop_revenue_per_visitor,
  ROUND(SUM(CASE WHEN main.event_name = 'purchase' AND main.device.category = 'desktop' THEN main.ecommerce.purchase_revenue ELSE 0 END) / NULLIF(COUNT(DISTINCT CASE WHEN main.event_name = 'purchase' AND main.device.category = 'desktop' THEN main.ecommerce.transaction_id END), 0), 2) AS desktop_average_order_value"""
    else:
        select_cols = """
  -- All devices
  COUNT(DISTINCT main.user_pseudo_id) AS total_visitors,
  COUNT(DISTINCT CASE WHEN main.event_name = 'purchase' THEN main.user_pseudo_id END) AS total_conversions,
  -- Mobile
  COUNT(DISTINCT CASE WHEN main.device.category = 'mobile' THEN main.user_pseudo_id END) AS mobile_visitors,
  COUNT(DISTINCT CASE WHEN main.event_name = 'purchase' AND main.device.category = 'mobile' THEN main.user_pseudo_id END) AS mobile_conversions,
  -- Desktop
  COUNT(DISTINCT CASE WHEN main.device.category = 'desktop' THEN main.user_pseudo_id END) AS desktop_visitors,
  COUNT(DISTINCT CASE WHEN main.event_name = 'purchase' AND main.device.category = 'desktop' THEN main.user_pseudo_id END) AS desktop_conversions"""
        if p.kpi_add_to_cart:
            select_cols += """,
  -- Add to cart (all devices / mobile / desktop)
  COUNT(DISTINCT CASE WHEN main.event_name = 'add_to_cart' THEN main.user_pseudo_id END) AS total_add_to_cart_conversions,
  COUNT(DISTINCT CASE WHEN main.event_name = 'add_to_cart' AND main.device.category = 'mobile' THEN main.user_pseudo_id END) AS mobile_add_to_cart_conversions,
  COUNT(DISTINCT CASE WHEN main.event_name = 'add_to_cart' AND main.device.category = 'desktop' THEN main.user_pseudo_id END) AS desktop_add_to_cart_conversions"""

    return f"""-- Baseline export ({p.output_type}, aggregate) — sample size preparation
DECLARE start_date STRING DEFAULT '{p.start_date}';
DECLARE end_date   STRING DEFAULT '{p.end_date}';

WITH{view_page_cte}
dummy AS (SELECT 1)  -- placeholder when no page filter

SELECT{select_cols}
FROM {table} AS main
{join_clause}
WHERE main.{suffix}{limit_clause};
"""


def _build_baseline_daily(p: BaselineParams, limit: int = 0) -> str:
    table = _table_ref(p.project, p.dataset)
    suffix = _suffix_filter(p.start_date, p.end_date)
    view_page_cte, join_clause = _baseline_page_filter(p, table, suffix)
    limit_clause = f"\nLIMIT {limit}" if limit else ""

    if p.output_type == "revenue":
        select_cols = """
  PARSE_DATE('%Y%m%d', main.event_date) AS report_date,
  COUNT(DISTINCT main.user_pseudo_id) AS visitors,
  COUNT(DISTINCT CASE WHEN main.event_name = 'purchase' THEN main.ecommerce.transaction_id END) AS transactions,
  SUM(CASE WHEN main.event_name = 'purchase' THEN main.ecommerce.purchase_revenue ELSE 0 END) AS purchase_revenue"""
    else:
        select_cols = """
  PARSE_DATE('%Y%m%d', main.event_date) AS report_date,
  COUNT(DISTINCT main.user_pseudo_id) AS visitors,
  COUNT(DISTINCT CASE WHEN main.event_name = 'purchase' THEN main.user_pseudo_id END) AS conversions"""
        if p.kpi_add_to_cart:
            select_cols += """,
  COUNT(DISTINCT CASE WHEN main.event_name = 'add_to_cart' THEN main.user_pseudo_id END) AS add_to_cart_conversions"""

    return f"""-- Baseline export ({p.output_type}, daily rows) — sample size preparation
DECLARE start_date STRING DEFAULT '{p.start_date}';
DECLARE end_date   STRING DEFAULT '{p.end_date}';

WITH{view_page_cte}
dummy AS (SELECT 1)  -- placeholder when no page filter

SELECT{select_cols}
FROM {table} AS main
{join_clause}
WHERE main.{suffix}
GROUP BY report_date
ORDER BY report_date{limit_clause};
"""


def _build_baseline_per_user(p: BaselineParams, limit: int = 0) -> str:
    """One row per completed order — for fitting a distribution (e.g. Negative
    Binomial or Gamma) from raw data in the pre-test tool's continuous KPI mode.
    Revenue-only: a binomial baseline has no per-order raw value to fit."""
    table = _table_ref(p.project, p.dataset)
    suffix = _suffix_filter(p.start_date, p.end_date)
    view_page_cte, join_clause = _baseline_page_filter(p, table, suffix)
    limit_clause = f"\nLIMIT {limit}" if limit else ""

    return f"""-- Baseline export (revenue, per-order raw values) — sample size preparation
DECLARE start_date STRING DEFAULT '{p.start_date}';
DECLARE end_date   STRING DEFAULT '{p.end_date}';

WITH{view_page_cte}
dummy AS (SELECT 1)  -- placeholder when no page filter

SELECT
  main.user_pseudo_id AS user_pseudo_id,
  main.ecommerce.transaction_id AS transaction_id,
  main.ecommerce.purchase_revenue AS purchase_revenue,
  main.ecommerce.total_item_quantity AS total_item_quantity
FROM {table} AS main
{join_clause}
WHERE main.{suffix}
  AND main.event_name = 'purchase'
  AND main.ecommerce.purchase_revenue IS NOT NULL
  AND main.ecommerce.purchase_revenue <> 0.0{limit_clause};
"""


# ---------------------------------------------------------------------------
# BINOMIAL
# ---------------------------------------------------------------------------

def _variant_case_block(exp: ExperimentConfig, strategy: str, source_col: str = "params.value.string_value") -> str:
    """Builds the CASE WHEN block mapping variant strings to labels."""
    lines = []
    for v in exp.variants:
        if strategy == "exact":
            lines.append(f"      WHEN {source_col} = '{v.string}' THEN '{v.label}'")
        else:
            lines.append(f"      WHEN {source_col} LIKE '%{v.string}%' THEN '{v.label}'")
    return "\n".join(lines)


def build_binomial(p: BinomialParams, limit: int = 0) -> str:
    table  = _table_ref(p.project, p.dataset)
    suffix = _suffix_filter(p.start_date, p.end_date)
    suffix_e = _suffix_filter(p.start_date, p.end_date, alias="e")
    exp    = p.experiments[0]

    all_variant_strings = [v.string for v in exp.variants]
    variant_in_list = ", ".join(f"'{s}'" for s in all_variant_strings)

    if p.match_strategy == "like":
        exp_filter = f"AND params.value.string_value LIKE '%{exp.prefix}%'"
    else:
        exp_filter = f"AND params.value.string_value IN ({variant_in_list})"

    # CASE WHEN in variant_data operates on exp_variant_string (extracted from
    # user_initial_exposure), not directly on params.value.string_value.
    case_block = _variant_case_block(exp, p.match_strategy, source_col="exp_variant_string")

    # -----------------------------------------------------------------------
    # Exposure CTEs
    # Always use user_initial_exposure + ROW_NUMBER to guarantee one row per
    # user (deduplicates repeated exposures, preventing fan-out in downstream
    # joins). When post_exposure_filter is on, first_exposure_timestamp is
    # carried into variant_data so event CTEs can filter against it.
    # -----------------------------------------------------------------------
    ts_col = "event_timestamp AS first_exposure_timestamp," if p.post_exposure_filter else ""

    exposure_ctes = f"""
user_initial_exposure AS (
  SELECT
    user_pseudo_id,
    params.value.string_value AS exp_variant_string,
    event_timestamp,
    ROW_NUMBER() OVER (
      PARTITION BY user_pseudo_id
      ORDER BY event_timestamp ASC
    ) AS rn
  FROM {table}, UNNEST(event_params) AS params
  WHERE {suffix}
    AND params.key = '{p.param_key}'
    AND params.value.string_value IS NOT NULL
    {exp_filter}
),

variant_data AS (
  SELECT
    user_pseudo_id AS variant_user_pseudo_id,
    {ts_col}
    CASE
{case_block}
      ELSE 'Other'
    END AS experience_variant_label
  FROM user_initial_exposure
  WHERE rn = 1
),
"""

    # -----------------------------------------------------------------------
    # Conditional CTEs — ecommerce, device, optional KPIs
    # When post_exposure_filter is on: event CTEs join variant_data to filter
    # purchases/events to those occurring after first exposure.
    # device_data is exempt — device category is a user attribute, not a
    # timestamped event.
    # -----------------------------------------------------------------------
    ecommerce_cte  = ""
    ecommerce_join = ""
    device_cte     = ""
    device_join    = ""
    optional_ctes  = ""
    optional_joins = ""

    need_ecommerce = p.kpi_transactions or p.kpi_aov

    if need_ecommerce:
        if p.post_exposure_filter:
            ecommerce_cte = f"""
ecommerce_data AS (
  SELECT
    e.user_pseudo_id AS ecommerce_user_pseudo_id,
    SUM(e.ecommerce.purchase_revenue)          AS purchase_revenue,
    COUNT(DISTINCT e.ecommerce.transaction_id) AS transaction_id
  FROM {table} e
  INNER JOIN variant_data vd ON e.user_pseudo_id = vd.variant_user_pseudo_id
  WHERE {suffix_e}
    AND e.event_name = 'purchase'
    AND e.user_pseudo_id IS NOT NULL
    AND e.event_timestamp >= vd.first_exposure_timestamp
  GROUP BY e.user_pseudo_id
),
"""
        else:
            ecommerce_cte = f"""
ecommerce_data AS (
  SELECT
    user_pseudo_id AS ecommerce_user_pseudo_id,
    SUM(ecommerce.purchase_revenue)          AS purchase_revenue,
    COUNT(DISTINCT ecommerce.transaction_id) AS transaction_id
  FROM {table}
  WHERE {suffix}
    AND event_name = 'purchase'
    AND user_pseudo_id IS NOT NULL
  GROUP BY user_pseudo_id
),
"""
        ecommerce_join = "  LEFT JOIN ecommerce_data ed ON vd.variant_user_pseudo_id = ed.ecommerce_user_pseudo_id\n"

    if p.kpi_device_split:
        # Device is a user attribute — no post-exposure filter applied.
        device_cte = f"""
device_data AS (
  SELECT
    user_pseudo_id AS device_user_pseudo_id,
    MAX(CASE WHEN device.category = 'mobile'  THEN 1 ELSE 0 END) AS is_mobile_user,
    MAX(CASE WHEN device.category = 'desktop' THEN 1 ELSE 0 END) AS is_desktop_user
  FROM {table}
  WHERE {suffix}
  GROUP BY user_pseudo_id
),
"""
        device_join = "  LEFT JOIN device_data dd ON vd.variant_user_pseudo_id = dd.device_user_pseudo_id\n"

    if p.kpi_add_to_cart:
        if p.post_exposure_filter:
            optional_ctes += f"""
add_to_cart_data AS (
  SELECT e.user_pseudo_id AS atc_user_pseudo_id
  FROM {table} e
  INNER JOIN variant_data vd ON e.user_pseudo_id = vd.variant_user_pseudo_id
  WHERE {suffix_e}
    AND e.event_name = 'add_to_cart'
    AND e.user_pseudo_id IS NOT NULL
    AND e.event_timestamp >= vd.first_exposure_timestamp
  GROUP BY e.user_pseudo_id
),
"""
        else:
            optional_ctes += f"""
add_to_cart_data AS (
  SELECT user_pseudo_id AS atc_user_pseudo_id
  FROM {table}
  WHERE {suffix}
    AND event_name = 'add_to_cart'
    AND user_pseudo_id IS NOT NULL
  GROUP BY user_pseudo_id
),
"""
        optional_joins += "  LEFT JOIN add_to_cart_data atc ON vd.variant_user_pseudo_id = atc.atc_user_pseudo_id\n"

    if p.kpi_login:
        if p.post_exposure_filter:
            optional_ctes += f"""
login_data AS (
  SELECT e.user_pseudo_id AS login_user_pseudo_id, 1 AS has_logged_in
  FROM {table} e
  INNER JOIN variant_data vd ON e.user_pseudo_id = vd.variant_user_pseudo_id,
  UNNEST(e.event_params) AS ep
  WHERE {suffix_e}
    AND e.event_name = 'page_view'
    AND ep.key = 'page_location'
    AND ep.value.string_value LIKE '%/customer/account/login%'
    AND e.user_pseudo_id IS NOT NULL
    AND e.event_timestamp >= vd.first_exposure_timestamp
  GROUP BY 1
),
"""
        else:
            optional_ctes += f"""
login_data AS (
  SELECT user_pseudo_id AS login_user_pseudo_id, 1 AS has_logged_in
  FROM {table}, UNNEST(event_params) AS ep
  WHERE {suffix}
    AND event_name = 'page_view'
    AND ep.key = 'page_location'
    AND ep.value.string_value LIKE '%/customer/account/login%'
    AND user_pseudo_id IS NOT NULL
  GROUP BY 1
),
"""
        optional_joins += "  LEFT JOIN login_data ld ON vd.variant_user_pseudo_id = ld.login_user_pseudo_id\n"

    if p.kpi_create_account:
        if p.post_exposure_filter:
            optional_ctes += f"""
create_account_data AS (
  SELECT e.user_pseudo_id AS create_user_pseudo_id, 1 AS has_created_account
  FROM {table} e
  INNER JOIN variant_data vd ON e.user_pseudo_id = vd.variant_user_pseudo_id,
  UNNEST(e.event_params) AS ep
  WHERE {suffix_e}
    AND e.event_name = 'page_view'
    AND ep.key = 'page_location'
    AND ep.value.string_value LIKE '%/customer/account/register%'
    AND e.user_pseudo_id IS NOT NULL
    AND e.event_timestamp >= vd.first_exposure_timestamp
  GROUP BY 1
),
"""
        else:
            optional_ctes += f"""
create_account_data AS (
  SELECT user_pseudo_id AS create_user_pseudo_id, 1 AS has_created_account
  FROM {table}, UNNEST(event_params) AS ep
  WHERE {suffix}
    AND event_name = 'page_view'
    AND ep.key = 'page_location'
    AND ep.value.string_value LIKE '%/customer/account/register%'
    AND user_pseudo_id IS NOT NULL
  GROUP BY 1
),
"""
        optional_joins += "  LEFT JOIN create_account_data cd ON vd.variant_user_pseudo_id = cd.create_user_pseudo_id\n"

    if p.kpi_ideal:
        if p.post_exposure_filter:
            optional_ctes += f"""
ideal_users AS (
  SELECT DISTINCT e.user_pseudo_id
  FROM {table} e
  INNER JOIN variant_data vd ON e.user_pseudo_id = vd.variant_user_pseudo_id,
  UNNEST(e.event_params) AS params
  WHERE {suffix_e}
    AND e.event_name = 'add_payment_info'
    AND params.key = 'payment_type'
    AND params.value.string_value LIKE '%iDEAL%'
    AND e.user_pseudo_id IS NOT NULL
    AND e.event_timestamp >= vd.first_exposure_timestamp
),
"""
        else:
            optional_ctes += f"""
ideal_users AS (
  SELECT DISTINCT user_pseudo_id
  FROM {table}, UNNEST(event_params) AS params
  WHERE {suffix}
    AND event_name = 'add_payment_info'
    AND params.key = 'payment_type'
    AND params.value.string_value LIKE '%iDEAL%'
    AND user_pseudo_id IS NOT NULL
),
"""
        optional_joins += "  LEFT JOIN ideal_users iu ON vd.variant_user_pseudo_id = iu.user_pseudo_id\n"

    # -----------------------------------------------------------------------
    # final_data SELECT list — only include columns whose CTEs are present.
    # -----------------------------------------------------------------------
    final_cols = [
        "    vd.variant_user_pseudo_id",
        "    vd.experience_variant_label",
    ]
    if need_ecommerce:
        final_cols += [
            "    ed.transaction_id  AS transaction_id",
            "    ed.purchase_revenue AS purchase_revenue",
        ]
    if p.kpi_device_split:
        final_cols += [
            "    dd.is_mobile_user",
            "    dd.is_desktop_user",
        ]
    if p.kpi_add_to_cart:
        final_cols.append(
            "    CASE WHEN atc.atc_user_pseudo_id IS NOT NULL THEN 1 ELSE 0 END AS has_added_to_cart"
        )
    if p.kpi_ideal:
        final_cols.append(
            "    CASE WHEN iu.user_pseudo_id IS NOT NULL THEN 1 ELSE 0 END AS paid_with_ideal"
        )
    if p.kpi_login:
        final_cols.append("    COALESCE(ld.has_logged_in, 0) AS has_logged_in")
    if p.kpi_create_account:
        final_cols.append("    COALESCE(cd.has_created_account, 0) AS has_created_account")

    final_select = ",\n".join(final_cols)

    # -----------------------------------------------------------------------
    # Outer aggregated SELECT — plain column names, no table alias prefixes.
    # -----------------------------------------------------------------------
    select_cols = [
        "  experience_variant_label",
        "  COUNT(DISTINCT variant_user_pseudo_id) AS visitors",
    ]
    if p.kpi_transactions:
        select_cols += [
            "  COUNT(DISTINCT CASE WHEN transaction_id IS NOT NULL THEN variant_user_pseudo_id END) AS users_with_transaction",
            "  SUM(CASE WHEN transaction_id IS NOT NULL THEN transaction_id ELSE 0 END) AS total_transactions",
        ]
    if p.kpi_aov:
        select_cols.append(
            "  ROUND(SUM(purchase_revenue) / NULLIF(SUM(CASE WHEN transaction_id IS NOT NULL THEN transaction_id ELSE 0 END), 0), 2) AS average_order_value"
        )
    if p.kpi_device_split:
        select_cols += [
            "  COUNT(DISTINCT CASE WHEN is_mobile_user  = 1 THEN variant_user_pseudo_id END) AS mobile_users",
            "  COUNT(DISTINCT CASE WHEN is_desktop_user = 1 THEN variant_user_pseudo_id END) AS desktop_users",
        ]
        if p.kpi_transactions:
            select_cols += [
                "  COUNT(DISTINCT CASE WHEN is_mobile_user  = 1 AND transaction_id IS NOT NULL THEN variant_user_pseudo_id END) AS mobile_buyers",
                "  COUNT(DISTINCT CASE WHEN is_desktop_user = 1 AND transaction_id IS NOT NULL THEN variant_user_pseudo_id END) AS desktop_buyers",
            ]
    if p.kpi_add_to_cart:
        select_cols.append("  SUM(has_added_to_cart) AS add_to_cart")
    if p.kpi_ideal:
        select_cols.append("  SUM(paid_with_ideal) AS paid_with_ideal")
    if p.kpi_login:
        select_cols.append("  SUM(has_logged_in) AS login_page_visits")
    if p.kpi_create_account:
        select_cols.append("  SUM(has_created_account) AS account_creation_page_visits")

    select_block = ",\n".join(select_cols)
    limit_clause = f"\nLIMIT {limit}" if limit else ""

    return f"""-- Binomial experiment export
DECLARE start_date STRING DEFAULT '{p.start_date}';
DECLARE end_date   STRING DEFAULT '{p.end_date}';

WITH
{exposure_ctes}{ecommerce_cte}{device_cte}{optional_ctes}
final_data AS (
  SELECT
{final_select}
  FROM variant_data vd
{ecommerce_join}{device_join}{optional_joins}  WHERE vd.experience_variant_label != 'Other'
)

SELECT
{select_block}
FROM final_data
GROUP BY experience_variant_label
ORDER BY experience_variant_label{limit_clause};
"""


# ---------------------------------------------------------------------------
# CONTINUOUS
# ---------------------------------------------------------------------------

def build_continuous(p: ContinuousParams, limit: int = 0) -> str:
    table = _table_ref(p.project, p.dataset)
    suffix = _suffix_filter(p.start_date, p.end_date)
    exp = p.experiments[0]

    all_variant_strings = [v.string for v in exp.variants]
    variant_in_list = ", ".join(f"'{s}'" for s in all_variant_strings)

    if p.match_strategy == "like":
        exp_filter = f"AND exp_variant_string LIKE '%{exp.prefix}%'"
    else:
        exp_filter = f"AND exp_variant_string IN ({variant_in_list})"

    case_lines = []
    for v in exp.variants:
        op = "LIKE" if p.match_strategy == "like" else "="
        case_lines.append(f"      WHEN exp_variant_string {op} '{v.string}' THEN '{v.label}'")
    case_block = "\n".join(case_lines)

    device_where = "" if p.device_filter == "all" else f"AND dd.primary_device = '{p.device_filter}'"

    join_type = "INNER JOIN" if p.query_mode == "revenue_only" else "LEFT JOIN"

    limit_clause = f"\nLIMIT {limit}" if limit else ""

    return f"""-- Continuous experiment export
DECLARE start_date    STRING DEFAULT '{p.start_date}';
DECLARE end_date      STRING DEFAULT '{p.end_date}';
DECLARE device_filter STRING DEFAULT '{p.device_filter}';

WITH
single_scan AS (
  SELECT
    user_pseudo_id,
    event_name,
    event_timestamp,
    ecommerce,
    (SELECT p.value.string_value
     FROM UNNEST(event_params) AS p
     WHERE p.key = '{p.param_key}'
     LIMIT 1) AS exp_variant_string
  FROM {table}
  WHERE {suffix}
    AND (
      event_name = 'purchase'
      OR EXISTS (SELECT 1 FROM UNNEST(event_params) AS p WHERE p.key = '{p.param_key}')
    )
    AND user_pseudo_id IS NOT NULL
),

user_initial_exposure AS (
  SELECT
    user_pseudo_id,
    exp_variant_string,
    ROW_NUMBER() OVER (PARTITION BY user_pseudo_id ORDER BY event_timestamp ASC) AS rn
  FROM single_scan
  WHERE exp_variant_string IS NOT NULL
    {exp_filter}
),

variant_data AS (
  SELECT
    user_pseudo_id,
    CASE
{case_block}
      ELSE 'Other'
    END AS experience_variant_label
  FROM user_initial_exposure
  WHERE rn = 1
    AND CASE
{case_block}
          ELSE 'Other'
        END != 'Other'
),

device_data AS (
  SELECT
    user_pseudo_id AS device_user_pseudo_id,
    CASE
      WHEN COUNTIF(device.category = 'desktop') >= COUNTIF(device.category = 'mobile') THEN 'desktop'
      ELSE 'mobile'
    END AS primary_device
  FROM {table}
  WHERE {suffix}
    AND user_pseudo_id IS NOT NULL
    AND device.category IN ('desktop', 'mobile')
  GROUP BY user_pseudo_id
),

ecommerce_data AS (
  SELECT
    user_pseudo_id,
    ecommerce.transaction_id AS transaction_id,
    SUM(ecommerce.purchase_revenue)    AS purchase_revenue,
    SUM(ecommerce.total_item_quantity) AS total_item_quantity
  FROM single_scan
  WHERE event_name = 'purchase'
    AND ecommerce.purchase_revenue IS NOT NULL
    AND ecommerce.purchase_revenue <> 0.0
  GROUP BY user_pseudo_id, transaction_id
)

SELECT
  vd.user_pseudo_id        AS variant_user_pseudo_id,
  vd.experience_variant_label,
  ed.purchase_revenue,
  ed.total_item_quantity,
  ed.transaction_id
FROM variant_data vd
{join_type} ecommerce_data ed ON vd.user_pseudo_id = ed.user_pseudo_id
INNER JOIN device_data      dd ON vd.user_pseudo_id = dd.device_user_pseudo_id
WHERE ('{p.device_filter}' = 'all' OR dd.primary_device = '{p.device_filter}')
ORDER BY ed.purchase_revenue DESC{limit_clause};
"""


# ---------------------------------------------------------------------------
# SEQUENTIAL
# ---------------------------------------------------------------------------

def build_sequential(p: SequentialParams, limit: int = 0) -> str:
    table = _table_ref(p.project, p.dataset)
    suffix = _suffix_filter(p.start_date, p.end_date)
    exp = p.experiments[0]

    all_variant_strings = [v.string for v in exp.variants]
    variant_in_list = ", ".join(f"'{s}'" for s in all_variant_strings)

    case_lines = []
    for v in exp.variants:
        case_lines.append(f"      WHEN exp_variant_string = '{v.string}' THEN '{v.label}'")
    case_block = "\n".join(case_lines)

    cumulative_table = f"`{p.cumulative_table}`" if p.cumulative_table else "`project.dataset.cumulative_test_data`"

    persistence_create = ""
    persistence_reset = ""
    persistence_insert = ""
    persistence_flag = str(p.use_persistence).upper()
    reset_flag = str(p.reset_cumulative_data).upper()

    optional_cols_extract = ""
    optional_cols_schema = ""
    if p.kpi_ideal:
        optional_cols_schema += "  paid_with_ideal INT64,\n"
        optional_cols_extract += "  CASE WHEN iu.user_pseudo_id IS NOT NULL THEN 1 ELSE 0 END AS paid_with_ideal,\n"

    if p.kpi_login:
        optional_cols_schema += "  has_logged_in INT64,\n"
        optional_cols_extract += "  COALESCE(ld.has_logged_in, 0) AS has_logged_in,\n"

    final_selects = ["  experience_variant_label", "  COUNT(DISTINCT variant_user_pseudo_id) AS visitors"]
    if p.kpi_transactions:
        final_selects += [
            "  COUNT(DISTINCT CASE WHEN transaction_id IS NOT NULL THEN variant_user_pseudo_id END) AS users_with_transaction",
            "  SUM(transaction_id) AS total_transactions",
        ]
    if p.kpi_aov:
        final_selects.append(
            "  ROUND(SUM(purchase_revenue) / NULLIF(COUNT(DISTINCT CASE WHEN transaction_id IS NOT NULL THEN variant_user_pseudo_id END), 0), 2) AS average_order_value"
        )
    if p.kpi_device_split:
        final_selects += [
            "  COUNT(DISTINCT CASE WHEN is_mobile_user  = 1 THEN variant_user_pseudo_id END) AS mobile_users",
            "  COUNT(DISTINCT CASE WHEN is_desktop_user = 1 THEN variant_user_pseudo_id END) AS desktop_users",
        ]
    if p.kpi_add_to_cart:
        final_selects.append("  COUNT(DISTINCT CASE WHEN added_to_cart = 1 THEN variant_user_pseudo_id END) AS add_to_cart")
    if p.kpi_ideal:
        final_selects.append("  COUNT(DISTINCT CASE WHEN paid_with_ideal = 1 THEN variant_user_pseudo_id END) AS paid_with_ideal")

    select_block = ",\n".join(final_selects)
    limit_clause = f"\nLIMIT {limit}" if limit else ""

    return f"""-- Sequential test export
DECLARE start_date           STRING DEFAULT '{p.start_date}';
DECLARE end_date             STRING DEFAULT '{p.end_date}';
DECLARE use_persistence      BOOL   DEFAULT {persistence_flag};
DECLARE reset_cumulative_data BOOL  DEFAULT {reset_flag};

--------------------------------------------------------------------------------
-- 1. TABLE MANAGEMENT
--------------------------------------------------------------------------------
IF use_persistence THEN
  CREATE TABLE IF NOT EXISTS {cumulative_table} (
    variant_user_pseudo_id STRING,
    experience_variant_label STRING,
    purchase_revenue FLOAT64,
    total_item_quantity INT64,
    transaction_id INT64,
    is_mobile_user INT64,
    is_desktop_user INT64,
    added_to_cart INT64,
{optional_cols_schema}    processed_at TIMESTAMP
  );
  IF reset_cumulative_data THEN
    TRUNCATE TABLE {cumulative_table};
  END IF;
END IF;

--------------------------------------------------------------------------------
-- 2. DATA EXTRACTION
--------------------------------------------------------------------------------
CREATE TEMP TABLE combined_new_data AS

WITH
user_initial_exposure AS (
  SELECT
    user_pseudo_id,
    params.value.string_value AS exp_variant_string,
    ROW_NUMBER() OVER (PARTITION BY user_pseudo_id ORDER BY event_timestamp ASC) AS rn
  FROM {table}, UNNEST(event_params) AS params
  WHERE {suffix}
    AND params.key = '{p.param_key}'
    AND params.value.string_value IN ({variant_in_list})
    AND user_pseudo_id IS NOT NULL
),

variant_data AS (
  SELECT
    user_pseudo_id AS variant_user_pseudo_id,
    CASE
{case_block}
      ELSE 'Other'
    END AS experience_variant_label
  FROM user_initial_exposure
  WHERE rn = 1
),

ecommerce_data AS (
  SELECT
    user_pseudo_id AS ecommerce_user_pseudo_id,
    SUM(ecommerce.purchase_revenue)          AS purchase_revenue,
    SUM(ecommerce.total_item_quantity)       AS total_item_quantity,
    COUNT(DISTINCT ecommerce.transaction_id) AS transaction_id
  FROM {table}
  WHERE {suffix}
    AND event_name = 'purchase'
    AND user_pseudo_id IS NOT NULL
  GROUP BY 1
),

add_to_cart_data AS (
  SELECT user_pseudo_id AS atc_user_pseudo_id
  FROM {table}
  WHERE {suffix}
    AND event_name = 'add_to_cart'
    AND user_pseudo_id IS NOT NULL
  GROUP BY 1
),

device_data AS (
  SELECT
    user_pseudo_id AS device_user_pseudo_id,
    MAX(CASE WHEN device.category = 'mobile'  THEN 1 ELSE 0 END) AS is_mobile_user,
    MAX(CASE WHEN device.category = 'desktop' THEN 1 ELSE 0 END) AS is_desktop_user
  FROM {table}
  WHERE {suffix}
  GROUP BY 1
)

SELECT
  vd.variant_user_pseudo_id,
  vd.experience_variant_label,
  COALESCE(ed.purchase_revenue, 0)    AS purchase_revenue,
  COALESCE(ed.total_item_quantity, 0) AS total_item_quantity,
  ed.transaction_id,
  COALESCE(dd.is_mobile_user, 0)      AS is_mobile_user,
  COALESCE(dd.is_desktop_user, 0)     AS is_desktop_user,
  CASE WHEN atc.atc_user_pseudo_id IS NOT NULL THEN 1 ELSE 0 END AS added_to_cart,
{optional_cols_extract}  CURRENT_TIMESTAMP() AS processed_at
FROM variant_data vd
LEFT JOIN ecommerce_data   ed  ON vd.variant_user_pseudo_id = ed.ecommerce_user_pseudo_id
LEFT JOIN device_data      dd  ON vd.variant_user_pseudo_id = dd.device_user_pseudo_id
LEFT JOIN add_to_cart_data atc ON vd.variant_user_pseudo_id = atc.atc_user_pseudo_id
WHERE vd.experience_variant_label != 'Other';

--------------------------------------------------------------------------------
-- 3. CUMULATIVE UPDATE
--------------------------------------------------------------------------------
IF use_persistence THEN
  INSERT INTO {cumulative_table}
  SELECT * FROM combined_new_data
  WHERE variant_user_pseudo_id NOT IN (
    SELECT variant_user_pseudo_id FROM {cumulative_table}
  );
END IF;

--------------------------------------------------------------------------------
-- 4. FINAL AGGREGATION
--------------------------------------------------------------------------------
WITH final_source AS (
  SELECT * FROM combined_new_data       WHERE NOT use_persistence
  UNION ALL
  SELECT * FROM {cumulative_table} WHERE use_persistence
)

SELECT
{select_block}
FROM final_source
GROUP BY 1
ORDER BY 1{limit_clause};
"""


# ---------------------------------------------------------------------------
# INTERACTION
# ---------------------------------------------------------------------------

def build_interaction(p: InteractionParams, limit: int = 0) -> str:
    table = _table_ref(p.project, p.dataset)
    suffix = _suffix_filter(p.start_date, p.end_date)
    n = len(p.experiments)

    # Collect all variant strings for the IN filter
    all_strings = []
    for exp in p.experiments:
        for v in exp.variants:
            if v.string:
                all_strings.append(f"NULLIF('{v.string}', '')")
    in_clause = ", ".join(all_strings)

    # Build CONCAT CASE WHEN for each experiment
    concat_cases = []
    for exp in p.experiments:
        a_str = next((v.string for v in exp.variants if v.label == "A"), "")
        b_str = next((v.string for v in exp.variants if v.label == "B"), "")
        concat_cases.append(
            f"      CASE\n"
            f"        WHEN '{a_str}' = '' AND '{b_str}' = '' THEN ''\n"
            f"        WHEN '{a_str}' IN UNNEST(seen_variants) THEN 'A'\n"
            f"        WHEN '{b_str}' IN UNNEST(seen_variants) THEN 'B'\n"
            f"        ELSE '_'\n"
            f"      END"
        )
    concat_block = ",\n".join(concat_cases)

    select_cols = ["  experience_variant_label", "  COUNT(DISTINCT variant_user_pseudo_id) AS total_users"]
    if p.kpi_transactions:
        select_cols += [
            "  COUNT(DISTINCT CASE WHEN ed.transaction_id IS NOT NULL THEN cu.variant_user_pseudo_id END) AS users_with_transaction",
            "  COUNT(DISTINCT ed.transaction_id) AS total_transactions",
        ]
    if p.kpi_add_to_cart:
        select_cols.append("  COUNT(DISTINCT CASE WHEN atc.atc_user_pseudo_id IS NOT NULL THEN cu.variant_user_pseudo_id END) AS add_to_cart")

    select_block = ",\n".join(select_cols)
    limit_clause = f"\nLIMIT {limit}" if limit else ""

    ecommerce_cte = ""
    ecommerce_join = ""
    atc_cte = ""
    atc_join = ""

    if p.kpi_transactions:
        ecommerce_cte = f"""
ecommerce_data AS (
  SELECT
    user_pseudo_id AS ecommerce_user_pseudo_id,
    SUM(ecommerce.purchase_revenue)          AS purchase_revenue,
    COUNT(DISTINCT ecommerce.transaction_id) AS transaction_id
  FROM {table}
  WHERE {suffix}
    AND event_name = 'purchase'
  GROUP BY user_pseudo_id, ecommerce.transaction_id
),
"""
        ecommerce_join = "  LEFT JOIN ecommerce_data ed ON cu.variant_user_pseudo_id = ed.ecommerce_user_pseudo_id\n"

    if p.kpi_add_to_cart:
        atc_cte = f"""
add_to_cart_data AS (
  SELECT user_pseudo_id AS atc_user_pseudo_id
  FROM {table}
  WHERE {suffix}
    AND event_name = 'add_to_cart'
  GROUP BY user_pseudo_id
),
"""
        atc_join = "  LEFT JOIN add_to_cart_data atc ON cu.variant_user_pseudo_id = atc.atc_user_pseudo_id\n"

    return f"""-- Interaction export — {n} experiment(s)
DECLARE start_date         STRING DEFAULT '{p.start_date}';
DECLARE end_date           STRING DEFAULT '{p.end_date}';
DECLARE event_parameter_key STRING DEFAULT '{p.param_key}';

WITH
variant_data AS (
  SELECT
    user_pseudo_id AS variant_user_pseudo_id,
    ARRAY_AGG(DISTINCT params.value.string_value) AS seen_variants
  FROM {table}, UNNEST(event_params) AS params
  WHERE {suffix}
    AND params.key = event_parameter_key
    AND params.value.string_value IN ({in_clause})
  GROUP BY user_pseudo_id
),

classified_users AS (
  SELECT
    variant_user_pseudo_id,
    CONCAT(
{concat_block}
    ) AS experience_variant_label
  FROM variant_data
),
{ecommerce_cte}{atc_cte}
aggregated_data AS (
  SELECT
{select_block}
  FROM classified_users cu
{ecommerce_join}{atc_join}  WHERE experience_variant_label != ''
  GROUP BY experience_variant_label
)

SELECT * FROM aggregated_data
ORDER BY experience_variant_label{limit_clause};
"""


# ---------------------------------------------------------------------------
# AUTO-DETECT QUERIES
# ---------------------------------------------------------------------------

def build_autodetect_variants_query(
    project: str,
    dataset: str,
    start_date: str,
    end_date: str,
    param_key: str,
    prefix: str,
) -> str:
    table = _table_ref(project, dataset)
    suffix = _suffix_filter(start_date, end_date)
    return f"""
SELECT DISTINCT params.value.string_value AS variant_string
FROM {table}, UNNEST(event_params) AS params
WHERE {suffix}
  AND params.key = '{param_key}'
  AND params.value.string_value LIKE '%{prefix}%'
  AND params.value.string_value IS NOT NULL
ORDER BY variant_string
LIMIT 500;
"""


def build_autodetect_kpi_query(
    project: str,
    dataset: str,
    start_date: str,
    end_date: str,
) -> str:
    table = _table_ref(project, dataset)
    suffix = _suffix_filter(start_date, end_date)
    return f"""
SELECT DISTINCT event_name
FROM {table}
WHERE {suffix}
  AND event_name IN ('purchase', 'add_to_cart', 'add_payment_info', 'page_view')
ORDER BY event_name
LIMIT 50;
"""
