"""Demo: produce a balance-test LaTeX table from a synthetic dataset.

Takes the synthetic firm-level dataset from ``demonstration_data``, which splits
firms into two groups (AI adopters vs non adopters) via a binary
``hire_manager_year`` column, and renders the balance table to LaTeX using
``descriptive_tables``.

Run it from anywhere; the .tex files land in a ``tables/`` directory next to
this script. Pasted line by line into a REPL, they land under the current
working directory instead, since there is no script to sit next to.
"""

from pathlib import Path

from demonstration_data import df_test_data
from descriptive_tables import balance_table, to_latex, descriptive_table, classify_variables

try:
    # Running as a script: put the output next to this file, whatever the cwd.
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    # No __file__ in a REPL / notebook / `python -c`: fall back to the cwd.
    SCRIPT_DIR = Path.cwd()

OUT_DIR = SCRIPT_DIR / "tables"

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
    df_test_data,
    var_list,
    categorical_threshold=20,
)


# ── Balance table ────────────────────────────────────────────────────────────
df_summary = balance_table(
    df_test_data,
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
    df_test_data,
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


