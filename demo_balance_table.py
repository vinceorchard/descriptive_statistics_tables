"""Demo: generate a fake dataset and produce a balance-test LaTeX table.

Builds a synthetic firm-level dataset with continuous, binary and categorical
variables, splits firms into two groups (AI adopters vs non adopters) via a
binary ``hire_manager_year`` column, and renders the balance table to LaTeX
using ``descriptive_tables``.

Run it from anywhere; the .tex files land in a ``tables/`` directory next to
this script.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from descriptive_tables import balance_table, to_latex, descriptive_table, classify_variables

OUT_DIR = Path(__name__).resolve().parent / "tables"

# ── Reproducible fake dataset ────────────────────────────────────────────────
rng = np.random.default_rng(42)
n = 1_000

# Group indicator: 1 = AI adopter, 0 = non adopter (also the OLS treatment).
hire_manager = rng.binomial(1, 0.45, n)

df = pd.DataFrame({
    "hire_manager_year": hire_manager,

    # Continuous variables (adopters slightly larger / more productive).
    "log_operating_revenue": rng.normal(8.0 + 0.4 * hire_manager, 1.1, n),
    "log_added_value": rng.normal(6.5 + 0.3 * hire_manager, 1.0, n),
    "log_revenue_per_employee": rng.normal(4.2 + 0.15 * hire_manager, 0.6, n),

    # Binary variables (adopters more likely to hold fundraising).
    "has_fundraising": rng.binomial(1, 0.25 + 0.15 * hire_manager, n),
    "has_new_fundraising": rng.binomial(1, 0.10 + 0.08 * hire_manager, n),

    # Categorical variables.
    "sector": rng.choice(
        ["Manufacturing", "Services & retail", "Finance", "ICT"],
        size=n,
        p=[0.30, 0.35, 0.15, 0.20],
    ),
    "age_cat": rng.choice(
        ["Young (<5y)", "Mature (5-20y)", "Old (>20y)"],
        size=n,
        p=[0.25, 0.50, 0.25],
    ),
})

# ── Variable lists by type (inferred from the number of distinct values) ─────
var_list = [
    "log_operating_revenue",
    "log_added_value",
    "log_revenue_per_employee",
    "has_fundraising",
    "has_new_fundraising",
    "sector",
    "age_cat",
]

binary_vars, categorical_vars, continuous_vars = classify_variables(
    df,
    var_list,
    categorical_threshold=20,
)


# ── Balance table ────────────────────────────────────────────────────────────
df_summary = balance_table(
    df,
    binary_vars=binary_vars,
    categorical_vars=categorical_vars,
    continuous_vars=continuous_vars,
    group_var="hire_manager_year",
    group_labels={1: "AI Adopters", 0: "Non Adopters"},
)

print(df_summary)

# ── Render to LaTeX ──────────────────────────────────────────────────────────
# No caption/label on purpose: passing them makes pandas wrap the output in a
# \begin{table} float. We want a bare tabular to \input{} into our own float.
latex = to_latex(df_summary)

OUT_DIR.mkdir(exist_ok=True)
out_path = OUT_DIR / "demo_balance_table.tex"
out_path.write_text(latex)

print(f"\nLaTeX table written to {out_path}")

# ── Descriptive table over ALL firms ─────────────────────────────────────────
df_summary = descriptive_table(
    df,
    binary_vars=binary_vars,
    categorical_vars=categorical_vars,
    continuous_vars=continuous_vars,
)

print(df_summary)

# ── Render to LaTeX ──────────────────────────────────────────────────────────
latex = to_latex(df_summary)

OUT_DIR.mkdir(exist_ok=True)
out_path = OUT_DIR / "demo_descriptive_table.tex"
out_path.write_text(latex)

print(f"\nLaTeX table written to {out_path}")


