"""Reusable functions to build descriptive and balance-test tables and render them to LaTeX.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

# Same mapping as in codes/balanched_check.py
LATEX_SPECIAL_CHARS_REPLACEMENT = {
    "&": r"and",
    "_": r" ",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "\\": r"\textbackslash{}",
    "<": r"\textless{}",
    ">": r"\textgreater{}",
    "|": r"\textbar{}",
    "/": r"\slash{}",
    "'": r"\textquotesingle{}",
}

_TRANS = str.maketrans(LATEX_SPECIAL_CHARS_REPLACEMENT)

# Sentinel occupying the second level of the internal (variable, category)
# MultiIndex for rows that describe a variable as a whole rather than one of its
# categories: the header row of a categorical block, and every binary /
# continuous row. Chosen as an object no user category value can equal.
_HEADER_KEY = object()


def _clean_label(label):
    """Escape LaTeX special characters and capitalise, as in the original script."""
    return str(label).translate(_TRANS).capitalize()


def _row_label(var, cat):
    """Display label for an internal ``(variable, category)`` index key."""
    return _clean_label(var) if cat is _HEADER_KEY else _clean_label(cat)


def classify_variables(df, vars_list, categorical_threshold=20):
    """Infer variable types from the number of distinct values.

    Classification rules (NaN does not count as a distinct value):

    * Non-numeric columns (object / string / category dtype) -> categorical.
    * Numeric columns:
        - exactly 2 distinct values            -> binary
        - 3 .. ``categorical_threshold``        -> categorical
        - more than ``categorical_threshold``   -> continuous

    Parameters
    ----------
    df : pandas.DataFrame
        The data.
    vars_list : list[str]
        Variables to classify.
    categorical_threshold : int, default 20
        Upper bound (inclusive) of distinct values for a numeric variable to be
        treated as categorical rather than continuous.

    Returns
    -------
    tuple[list[str], list[str], list[str]]
        ``(binary_vars, categorical_vars, continuous_vars)``, each preserving
        the input order of ``vars_list``.
    """
    binary_vars, categorical_vars, continuous_vars = [], [], []

    for var in vars_list:
        n_unique = df[var].nunique(dropna=True)

        if not pd.api.types.is_numeric_dtype(df[var]):
            categorical_vars.append(var)
        elif n_unique == 2:
            binary_vars.append(var)
        elif n_unique <= categorical_threshold:
            categorical_vars.append(var)
        else:
            continuous_vars.append(var)

    return binary_vars, categorical_vars, continuous_vars


def _var_level_index(var_list):
    """``(variable, _HEADER_KEY)`` keys for one-row-per-variable blocks."""
    return pd.MultiIndex.from_tuples([(v, _HEADER_KEY) for v in var_list])


def _continuous_block(data, var_list, mean_col, std_col, count_col):
    """mean / std / count for continuous variables (one row per variable)."""
    if not var_list:
        return pd.DataFrame(columns=[mean_col, std_col, count_col])
    block = (
        data[var_list]
        .agg(["mean", "std", "count"])
        .T.rename(columns={"mean": mean_col, "std": std_col, "count": count_col})
    )
    block.index = _var_level_index(var_list)
    return block[[mean_col, std_col, count_col]]


def _binary_block(data, var_list, mean_col, std_col, count_col):
    """mean / count for binary variables (std left empty)."""
    if not var_list:
        return pd.DataFrame(columns=[mean_col, std_col, count_col])
    block = (
        data[var_list]
        .agg(["mean", "count"])
        .T.rename(columns={"mean": mean_col, "count": count_col})
    )
    block[std_col] = np.nan
    block.index = _var_level_index(var_list)
    return block[[mean_col, std_col, count_col]]


def _categorical_block(data, var, mean_col, std_col, count_col, categories=None):
    """Header row (count only) followed by one share row per category.

    Mirrors the original script: the header row is labelled with the variable
    name and carries the overall non-missing count; each category row reports
    its share (``value_counts(normalize=True)``) in the mean column.

    ``categories``, when given, fixes the row order and guarantees that every
    category appears even if absent from ``data`` (its share is then 0).

    Rows are keyed by the ``(variable, category)`` pair, with ``_HEADER_KEY``
    marking the header row. Keeping the raw variable name and raw category
    value in the index makes every row addressable without ambiguity, even when
    two variables share a category value or a value equals a variable name.
    """
    shares = data[var].value_counts(normalize=True)
    if categories is None:
        categories = list(shares.index)
    else:
        shares = shares.reindex(categories, fill_value=0.0)

    index = pd.MultiIndex.from_tuples(
        [(var, _HEADER_KEY)] + [(var, cat) for cat in categories]
    )
    block = pd.DataFrame(
        {
            mean_col: [np.nan] + list(shares.loc[categories]),
            std_col: np.nan,
            count_col: [data[var].agg("count")] + [np.nan] * len(categories),
        },
        index=index,
    )
    return block[[mean_col, std_col, count_col]]


def descriptive_table(
    df,
    binary_vars,
    categorical_vars,
    continuous_vars,
    mean_col="Mean",
    std_col="Std",
    count_col="N",
):
    """Descriptive statistics (mean / std / count) for a single sample.

    Parameters
    ----------
    df : pandas.DataFrame
        The data.
    binary_vars, categorical_vars, continuous_vars : list[str]
        Variable names by type.
    mean_col, std_col, count_col : str
        Output column labels.

    Returns
    -------
    pandas.DataFrame
        Rows ordered as: categorical blocks (header + shares), then binary,
        then continuous. Std is empty for binary and categorical rows; mean and
        std are empty for categorical header rows.
    """
    table = _keyed_table(
        df, binary_vars, categorical_vars, continuous_vars,
        mean_col, std_col, count_col,
    )
    return _flatten_index(table)


def _keyed_table(
    df, binary_vars, categorical_vars, continuous_vars,
    mean_col, std_col, count_col, categories=None,
):
    """Build a descriptive table indexed by ``(variable, category)`` keys.

    ``categories`` optionally maps a categorical variable to the category order
    to use, so that two samples can be built on identical rows.
    """
    categories = categories or {}
    blocks = []

    for var in categorical_vars:
        blocks.append(
            _categorical_block(
                df, var, mean_col, std_col, count_col, categories.get(var)
            )
        )

    blocks.append(_binary_block(df, binary_vars, mean_col, std_col, count_col))
    blocks.append(_continuous_block(df, continuous_vars, mean_col, std_col, count_col))

    table = pd.concat([b for b in blocks if not b.empty], axis=0)
    return table[[mean_col, std_col, count_col]]


def _flatten_index(table):
    """Replace the internal ``(variable, category)`` keys with display labels.

    Category rows render with their bare value, scoped by the header row printed
    just above them. Two categorical variables sharing a category value ("Yes")
    therefore yield two rows with the same label -- intended, since the header
    row disambiguates them for the reader.
    """
    table = table.copy()
    table.index = [_row_label(var, cat) for var, cat in table.index]
    return table


def _run_ols(data, y, treat, controls):
    """OLS: y ~ treat + controls. Returns (beta_treat, pvalue_treat)."""
    X = data[[treat] + list(controls)].copy()
    X = sm.add_constant(X, has_constant="add")
    model = sm.OLS(y, X, missing="drop").fit()
    return model.params[treat], model.pvalues[treat]


def _smd(series1, series0):
    """Standardised mean difference (Lipsey & Wilson 2001), pooled-SD scaled."""
    n1, n0 = series1.count(), series0.count()
    if n1 + n0 <= 2:
        return np.nan
    m1, m0 = series1.mean(), series0.mean()
    s1, s0 = series1.std(ddof=1), series0.std(ddof=1)
    sp = np.sqrt(((n1 - 1) * s1 ** 2 + (n0 - 1) * s0 ** 2) / (n1 + n0 - 2))
    return (m1 - m0) / sp if sp > 0 else np.nan


def balance_table(
    df,
    binary_vars,
    categorical_vars,
    continuous_vars,
    group_var,
    group_labels=None,
    controls=None,
    diff_col="Difference in means",
    pvalue_col="P-value",
    smd_col="Standardised Mean Difference",
    compute_smd=True,
):
    """Two-group descriptive statistics with a difference-in-means OLS test.

    The two groups are defined by ``df[group_var] == 1`` and
    ``df[group_var] == 0``. The same ``group_var`` is used as the treatment
    regressor in the OLS test ``variable ~ group_var (+ controls)``, exactly as
    in the original script.

    Parameters
    ----------
    df : pandas.DataFrame
        The data. ``group_var`` must be a binary 0/1 column.
    binary_vars, categorical_vars, continuous_vars : list[str]
        Variable names by type.
    group_var : str
        Binary column splitting the two groups and used as OLS treatment.
    group_labels : dict, optional
        Maps group value (1, 0) to a column label. Defaults to
        ``{1: "Group 1", 0: "Group 0"}``.
    controls : list[str], optional
        Additional OLS controls. Defaults to no controls (as in the script).
    diff_col, pvalue_col, smd_col : str
        Output column labels for the test results.
    compute_smd : bool, default True
        Whether to append the standardised mean difference column.

    Returns
    -------
    pandas.DataFrame
        MultiIndex columns ``(group_label, {Mean, Std, N})`` for each group,
        followed by single-level ``diff_col``, ``pvalue_col`` and, when
        ``compute_smd`` is true, ``smd_col``.

    Raises
    ------
    ValueError
        If ``group_var`` is not coded strictly as 0/1 with both groups present.
    """
    if group_labels is None:
        group_labels = {1: "Group 1", 0: "Group 0"}
    if controls is None:
        controls = []

    # The whole function assumes a 0/1 treatment: the group split below and the
    # OLS regressor both rely on it. Validate rather than silently emit a table
    # with an empty group.
    observed = set(pd.unique(df[group_var].dropna()))
    if not observed <= {0, 1}:
        raise ValueError(
            f"group_var {group_var!r} must be coded 0/1, found values "
            f"{sorted(observed, key=repr)}."
        )
    if observed != {0, 1}:
        raise ValueError(
            f"group_var {group_var!r} must contain both groups (0 and 1), "
            f"found only {sorted(observed, key=repr)}."
        )

    label1, label0 = group_labels[1], group_labels[0]
    df1 = df[df[group_var] == 1]
    df0 = df[df[group_var] == 0]

    # Fix the category order on the pooled sample so both group tables share the
    # same rows. A category seen only in one group therefore gets a 0 share in
    # the other (it is genuinely absent), not a missing value.
    categories = {
        var: list(df[var].value_counts().index) for var in categorical_vars
    }

    cols = ("Mean", "Std", "N")

    def _group_table(data):
        return _keyed_table(
            data, binary_vars, categorical_vars, continuous_vars,
            "Mean", "Std", "N", categories,
        )

    table1 = _group_table(df1)
    table0 = _group_table(df0)

    table1.columns = pd.MultiIndex.from_tuples([(label1, c) for c in cols])
    table0.columns = pd.MultiIndex.from_tuples([(label0, c) for c in cols])

    summary = pd.concat([table1, table0], axis=1)
    summary[diff_col] = np.nan
    summary[pvalue_col] = np.nan
    if compute_smd:
        summary[smd_col] = np.nan

    # Rows are keyed by (variable, category), so every assignment below targets
    # exactly one row even when two variables share a category value.
    # --- OLS difference-in-means tests (on the pooled sample) -----------------
    for var in list(continuous_vars) + list(binary_vars):
        beta, pval = _run_ols(df, df[var], group_var, controls)
        summary.loc[(var, _HEADER_KEY), diff_col] = beta
        summary.loc[(var, _HEADER_KEY), pvalue_col] = pval

    for cat in categorical_vars:
        dummies = pd.get_dummies(df[cat], drop_first=False) * 1
        pooled = pd.concat([df[[group_var] + list(controls)], dummies], axis=1)
        for raw_col in dummies.columns:
            beta, pval = _run_ols(pooled, pooled[raw_col], group_var, controls)
            summary.loc[(cat, raw_col), diff_col] = beta
            summary.loc[(cat, raw_col), pvalue_col] = pval

    # --- Standardised Mean Difference (Lipsey & Wilson 2001) -------------------
    if compute_smd:
        for var in list(continuous_vars) + list(binary_vars):
            summary.loc[(var, _HEADER_KEY), smd_col] = _smd(
                df1[var].dropna(), df0[var].dropna()
            )

        for cat in categorical_vars:
            dummies = pd.get_dummies(df[cat], drop_first=False) * 1
            for raw_col in dummies.columns:
                summary.loc[(cat, raw_col), smd_col] = _smd(
                    dummies.loc[df1.index, raw_col], dummies.loc[df0.index, raw_col]
                )

    return _flatten_index(summary)


def _fmt(x):
    """Integer-aware float formatter from the original script.

    Non-finite values (NaN, +/-inf) have no integer form and are rendered as an
    empty string rather than raising.
    """
    if not np.isfinite(x):
        return ""
    if x == int(x):
        return f"{int(x):,}"
    return f"{x:,.3f}"


def to_latex(df, **kwargs):
    """Render a produced table to a LaTeX string.

    Row/column labels are assumed to already be LaTeX-escaped by the table
    builders; this applies the integer-aware float format and the rendering
    options used in the original script. Extra keyword arguments (e.g.
    ``caption``, ``label``) are forwarded to ``DataFrame.to_latex``.
    """
    options = dict(
        float_format=_fmt,
        na_rep="",
        multicolumn=True,
        multicolumn_format="c",
        bold_rows=False,
    )
    options.update(kwargs)
    return df.to_latex(**options)
