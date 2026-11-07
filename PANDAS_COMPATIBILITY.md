# pandas 2 and 3 compatibility

Validation date: 2026-11-06. Branch: `chore-pandas-3`, based directly on
`1e0e7f5069be8f8b8b18e0a7b21dd7e543ef4783` (0.7.0). The initial working tree
was clean. The baseline is an ancestor of this branch; no cherry-picks were needed.
No repository AGENTS.md files or applicable ancestor instructions were present.

## Changes and supported dependencies

- pandas: `>=2.1.1,<4`; statsmodels: `>=0.14.6,<0.15`.
- `LogisticModel.inverse_function` and `simulate_decoys` now copy arrays before
  masking them. pandas 3 exports can be read-only under Copy-on-Write; both
  functions previously raised `ValueError: assignment destination is read-only`
  for Series inputs. Tests also pass explicitly read-only NumPy arrays and check
  that caller-owned inputs retain their values.
- statsmodels 0.14.5, previously locked, fails importing `statsmodels.api` on
  pandas 3.1.1: `deprecate_kwarg() missing 1 required positional argument:
  'new_arg_name'`. statsmodels 0.14.6 works through the actual BH adjustment path.
  Its metadata excludes pandas 2.1.0; the declared minimum is therefore 2.1.1.
- Bokeh's existing `>=3.4,<3.8` constraint is retained. Both 3.4.0 and 3.7.3
  generated dashboards on pandas 3.1.1. No parser, normalization, imputation,
  optimization, significance, or regulation formulas were changed.
- Poetry 2.1.3 regenerated `poetry.lock` without an unrelated dependency refresh:
  pandas stays 2.3.3 and statsmodels moves from 0.14.5 to 0.14.6.

These changes address pandas' documented
[Copy-on-Write behavior](https://pandas.pydata.org/docs/whatsnew/v3.0.0.html).
String inference and GroupBy usage were inspected; existing Spectronaut tests
cover TSV/Parquet, long/pivot reports, quantity selection, native PTM identities,
sequence collisions, duplicates, missing quantities/annotations and invalid run
labels. No parser changes were necessary for these covered contracts.

## Environments and checks

| Python | pandas | NumPy | SciPy | Bokeh | statsmodels | PyArrow | pytest |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3.13.5 | 2.3.3 | 2.5.4 | 1.18.1 | 3.7.3 | 0.14.6 | 26.0.1 | 9.2.0 |
| 3.13.5 | 3.1.1 | 2.5.4 | 1.18.1 | 3.7.3 | 0.14.6 | 26.0.1 | 9.2.0 |
| 3.13.5 | 3.0.6 | 2.5.4 | 1.18.1 | 3.7.3 | 0.14.6 | 26.0.1 | 9.2.0 |
| 3.11.15 | 2.1.1 | 1.26.4 | 1.16.3 | 3.7.3 | 0.14.6 | 21.0.0 | 9.2.0 |
| 3.13.5 | 2.3.3 (Poetry lock) | 2.3.2 | 1.16.1 | 3.7.3 | 0.14.6 | 26.0.1 | 7.4.4 |
| 3.13.5 | 3.1.1 + performance extras | 2.3.2 | 1.16.1 | 3.7.3 | 0.14.6 | 26.0.1 | 7.4.4 |

A separate synthetic dashboard run used Bokeh 3.4.0 with the pandas 3.1.1 row's
other dependencies. Core runs used tqdm 4.70.1, Patsy 1.0.3, Jinja2 3.1.6,
MarkupSafe 3.0.4 and Tornado 6.6. Full local environment inventories and logs are
in `/tmp/cc-*`; workflow runs write their own `versions.json`.

The unit suite passed 362 tests on pandas 2.3.3 and 3.1.1. The existing parser
suite alone passed 162 tests on 3.1.1. pandas 2.1.1 with PyArrow 21 and pandas
3.0.6 also passed all 362 tests and the synthetic workflow. For pandas 3.0.6,
all compared numerical outputs against pandas 2.3.3 were identical.

The Poetry installation and package build (wheel and sdist) succeeded, and
`poetry check --lock` succeeded with legacy metadata deprecation warnings.
The locked pandas 2 environment and pandas 3.1.1 with performance extras both
completed the synthetic workflow and passed the same numerical comparisons.
Performance extras were Bottleneck 1.6.0, NumExpr 2.14.2, Numba 0.68.1 and
llvmlite 0.50.0. `pip check` found no broken requirements. All 362 unit tests also passed in
this additional pandas 3 environment.

## Deterministic synthetic validation

`tests/compatibility/run_workflow.py` generates 24 protein identities across nine
doses, with duplicate precursor rows, deliberately missing genes and quantities,
a curve removed for missingness, imputation and median normalization enabled.
It snapshots protein, peptide and all three native PTM quantity selections,
fits curves, applies SAM and BH thresholds, generates seeded decoys, estimates
q-values, and renders a Bokeh dashboard. These are synthetic fixtures, including
the small parser examples; they are not independent biological validation.

With pandas 2.3.3 versus 3.1.1:

- Identical synthetic source files (432 rows). All 24 parsed protein identities,
  four peptide identities and two PTM identities per quantity selection agree.
  Missingness, annotations, row order and duplicate counts agree exactly.
- Fifteen target curves survive the configured missingness and imputation
  filters in both environments. SAM calls: four up, eleven unassigned. BH calls:
  seven up, eight unassigned. Calls agree exactly, including missing calls.
- Maximum target differences: pEC50 `3.56e-15`, slope `1.78e-15`, front `2.00e-15`,
  back `2.45e-15`, log2 fold change `4.67e-15`, p-value and BH adjusted p-value
  `3.61e-16`. q-values agree exactly. Normalized quantities differ by at most
  `4.37e-11`. The largest absolute target-output difference is `4.41e-7` in
  a large, poorly constrained slope standard error (about 31742); relative
  difference is approximately `1.4e-11`.
- The comparator checks **every** numeric column with `rtol=1e-8, atol=1e-10`,
  preserving exact missingness and categorical values. This allows numerical
  roundoff while demanding substantially more precision than reported biological
  results; near-zero p-values are also compared with this absolute tolerance.
- Unrounded empirical decoy inputs differed by at most `8.89e-16`. One nearly
  flat null fit amplified this into pEC50 difference `0.0737`, back plateau
  difference `0.6913`, and p-value difference `6.70e-7`; its regulation remained
  unassigned. Huge parameter uncertainties confirm weak identifiability.
  This is a remaining sensitivity of the existing fitter, not resolved by
  changing the scientific method here.
- To compare decoy fitting on identical numerical inputs, the validation script
  rounds only synthetic decoy intensities to 12 decimals before fitting. It
  separately retains and compares unrounded simulation outputs. With identical
  decoy inputs, all 15 fitted decoy outputs agree exactly. Production code does
  not perform this rounding. The small decoy count exercises mechanics, not
  statistically reliable FDR estimation.

## Real-data validation

The complete bundled `example_datasets/kinobeads_Dasatinib` dataset was copied
into separate temporary directories and run through the normal CLI with its
unchanged parameter file, including interpolation and imputation. Both pandas
2.3.3 and 3.1.1 runs completed and generated dashboards using Bokeh 3.7.3.

Both retained 1170 curves, with the same 47 output columns, identities,
missingness, fit parameters and errors, fold changes, p-values and relevance
scores. All numeric differences were **zero** after reading the saved TSVs.
Calls matched: 109 up, 66 down, 94 not regulated, 901 unassigned. This CLI run
used SAM without decoy FDR; real-data q-values were not validated.

## Reproduction and CI

In each installed environment, from the repository root:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest tests/unit
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. \
  python tests/compatibility/run_workflow.py /tmp/results-pandas-version
python tests/compatibility/compare_outputs.py /tmp/results-pandas2 /tmp/results-pandas3
```

For real data, copy `parameters.toml` and `proteinGroups.txt` from the bundled
Dasatinib directory into two temporary directories, run
`python -m curve_curator /tmp/directory/parameters.toml` in each environment,
then compare using `compare_outputs.py ... --pattern curves.txt`.

CI exercises Python 3.11/pandas 2.1.1, Python 3.12/pandas 2.3,
Python 3.13/pandas 2.3, 3.0 and 3.1 (with performance extras), and
Python 3.14/pandas 3.1.
Every job installs the package using Poetry, includes optional Parquet support,
checks dependency consistency, runs unit tests and the synthetic dashboard
workflow. A separate job compares Python 3.13 pandas 2 and 3 artifacts.
CI has been configured locally; hosted runs have not occurred.

## Limits and proposed Synapse follow-up

The initial pandas 2.1.1 / NumPy 1.26.4 run had 20 Parquet failures because
PyArrow 26 requires NumPy 2 at runtime. The minimum-version CI entry pins
PyArrow 21.0.0; users retaining NumPy 1 should also choose PyArrow 21.0.0.
This is a dependency combination issue outside the pandas 3 code fixes. pandas 3.1 emits deprecation warnings for existing `inplace`
operations; these concern pandas 4, outside the declared range. Existing warnings
also include SciPy and multiprocessing deprecations. Poetry check reports
existing legacy metadata warnings. Dashboard HTML generation was checked;
browser interaction was not automated. Other bundled biological datasets and
all combinations in the declared dependency ranges were not exhaustively tested.
pandas performance extras were exercised locally; the full Synapse application
environment was not validated.

After review and an upstream commit is available, a separate Synapse change can
pin that reviewed CurveCurator commit, install it into Synapse's Python 3.13 /
pandas[performance] 3.1 environment, run Synapse integration and dashboard checks,
and then remove the isolated environment. This CurveCurator PR makes no Synapse
changes and publishes no release.
