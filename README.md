# descriptive_statistics_tables

Build descriptive-statistics and balance-test tables from a pandas DataFrame and
render them to LaTeX.

Balance tables report, for each variable, the mean/std/N within each of two
groups, the difference in means and its p-value from an OLS regression
(`variable ~ group (+ controls)`), and the standardised mean difference.

## Install

There is no package to install — clone the repo and use the module directly:

```bash
git clone https://github.com/<your-username>/descriptive_statistics_tables.git
cd descriptive_statistics_tables
pip install -r requirements.txt
```

Then either work inside this directory, or copy `descriptive_tables.py` next to
your own code.

## Usage

The example below runs as-is: `demonstration_data` ships a synthetic firm-level
dataset, `df_test_data`. Swap it for your own DataFrame (`pd.read_csv(...)`) and
adjust the column names.

```python
from demonstration_data import df_test_data
from descriptive_tables import classify_variables, balance_table, to_latex

# Infer each variable's type from its number of distinct values.
binary, categorical, continuous = classify_variables(
    df_test_data,
    ["log_operating_revenue", "has_fundraising", "sector"],
    categorical_threshold=20,
)

table = balance_table(
    df_test_data, binary, categorical, continuous,
    group_var="hire_manager_year",                   # must be coded 0/1
    group_labels={1: "AI Adopters", 0: "Non Adopters"},
)

with open("balance.tex", "w") as f:
    f.write(to_latex(table))
```

For a table over the whole sample, use `descriptive_table(df_test_data, binary,
categorical, continuous)` instead.

Run `python demo_balance_table.py` for a complete worked example; it writes both
tables to `tables/`.

## The output

`to_latex` emits a bare `tabular` environment, so you can `\input{}` it into
your own float and control the caption, label and placement yourself:

```latex
\begin{table}[htbp]
  \centering
  \caption{Balance test}
  \resizebox{\textwidth}{!}{\input{balance.tex}}
\end{table}
```

Passing `caption=` or `label=` to `to_latex` makes pandas wrap the output in a
`\begin{table}` float instead.

Categorical variables render as a header row carrying the variable name and the
non-missing count, followed by one row per category giving its share. Category
rows show the bare category value; the header row above them identifies which
variable they belong to. Two categorical variables may therefore each contribute
a row labelled `Yes` — the header rows disambiguate them.

Row labels are escaped for LaTeX (`<5y` becomes `\textless{}5y`), so the output
compiles as-is.

## Notes

- `group_var` must be coded 0/1 and contain both groups; anything else raises
  `ValueError` rather than silently producing an empty group.
- A category present in only one group gets a share of 0 in the other, not a
  missing value.
- `pandas >= 3` renders `DataFrame.to_latex` through its Styler, which requires
  `jinja2`. It is listed in `requirements.txt`.

## Demo

`demo_balance_table.py` provides a minimal working example, using the synthetic
dataset in `demonstration_data.py`.

```bash
python demo_balance_table.py
```

## License

MIT — see [LICENSE](LICENSE).
