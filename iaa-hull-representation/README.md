# Convex Hull Representation Experiments

## Description

Exploratory Jupyter notebooks investigating how well a convex hull — sampled from archetypal analysis — can represent a whole dataset, and whether regression models fitted to a handful of hull samples can match models fitted to the full dataset.

The experiments aim at reducing the amount of data a regression model needs to see by exploiting the hull (or archetypes) as a compact representation, in the spirit of the [IterAA](https://github.com/Jon-Ting/iteraa) project.

## Contents

| Notebook | Purpose |
|---|---|
| `whole-data-vs-convex-hull-regression.ipynb` | Uses SPAMS archetypal analysis to obtain hull samples (via their closest real-data matches), then compares regressors (Ridge, Extra Trees, MLP, XGBoost) fitted on the whole data vs. the hull samples. Expects `dataForAndrew2.pickle` alongside it (not committed to the repository). |
| `linearity-impact-on-convex-hull-regression.ipynb` | Explores the impact of linearity on hull-sample regression: shows the positive results on the previous hull tests break down on the nonlinear surface `Z = sin(sqrt(X^2 + Y^2))`. |
| `convex-hull-representation-of-whole-data.ipynb` | Tests how well convex hulls represent extreme cases of a dataset, including linear and generated (`sklearn.datasets.make_regression`) data. |

## Environment

The notebooks share a `pyproject.toml`-defined environment, including `spams-bin` (which provides the `spams` package), matching the dependency floors used by the `iteraa` package.

```bash
uv sync
uv run jupyter lab
```

Alternatively:

```bash
pip install -e .[dev]
jupyter lab
```

## Notes

* The notebooks print the library versions of the run that produced their saved outputs for reproducibility; note the saved outputs span multiple library versions (e.g. `xgboost` 2.0.2 vs 3.4.1), so re-running the code against the floors in `pyproject.toml` may shift minor results.
* These are research notebooks — file layout, naming and results may shift as findings evolve.
