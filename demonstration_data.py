"""Synthetic firm-level dataset used by the demo and the documentation.

Exposes ``df_test_data``: a reproducible 1,000-row DataFrame with continuous,
binary and categorical variables, split into two groups by the binary
``hire_manager_year`` column (1 = AI adopter, 0 = non adopter). The adopters are
given slightly higher revenue and a higher fundraising probability, so the
balance table has something to detect.
"""

import numpy as np
import pandas as pd

# ── Reproducible fake dataset ────────────────────────────────────────────────
rng = np.random.default_rng(42)
n = 1_000

# Group indicator: 1 = AI adopter, 0 = non adopter (also the OLS treatment).
hire_manager = rng.binomial(1, 0.45, n)

df_test_data = pd.DataFrame({
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
