# Structure-Adaptive Estimation of Heterogeneous Treatment Effects with Kernel Methods

This repository contains code for the paper of the same title. The two notebooks
run the synthetic and RealCause experiments, including comparisons with Plug-in,
DR, Oracle DR, and R-learners.

## Problem Setting

We estimate the conditional average treatment effect

$$\tau(x) = \mathbb{E}[Y(1)-Y(0)\mid X=x].$$

The treatment effect can have simpler structure than the individual outcome
functions. Our estimator uses separate kernels for the outcomes and the treatment
effect to exploit this structure.

## Method Summary

1. Fit treatment-specific outcome functions by kernel ridge regression (KRR).
2. Construct imputed treatment effects and fit a second-stage KRR over candidate
   treatment-effect kernels and regularization parameters.
3. Select a candidate using a separate validation score and average predictions
   over three cyclic data splits.

The method is labeled **Ours** throughout the experiments. It does not require
propensity score estimation. Oracle DR uses the true propensity in synthetic
experiments and is omitted from RealCause, where that propensity is unavailable.

## Quick Start

From this folder, install the dependencies and start JupyterLab
(tested with Python 3.9.6):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyterlab
```

Open a notebook, edit **Settings**, and select **Run All**.

| Notebook | Small test run | Paper settings |
| --- | --- | --- |
| [Synthetic](UCATE_Synthetic_Demo.ipynb) | `PROFILE = "quick"` | `PROFILE = "full"`, `SETTING = "all"`, `STUDY = "all"`, `N_REPS = None` |
| [RealCause](UCATE_RealCause_Demo.ipynb) | `PROFILE = "quick"`, `N_SAMPLES = 2` | `PROFILE = "full"`, `N_SAMPLES = 50`, `START_SAMPLE = 0` |

Tables appear in the notebook. Set `PLOTS = True` to also display figures.
Results and checkpoints are saved under `outputs/`; matching runs resume
automatically. Saved paper results are available in [paper_results](paper_results/README.md).

## Experiment Options

- **Synthetic:** `SETTING` selects `univariate`, `multivariate`, or `all`.
  `STUDY` selects `sample_size`, `noise`, `overlap`, `nuisance`, `representation`,
  or `crossfit`; `all` runs these six studies. The additional `undersmoothing`
  and `same_kernel` studies can be selected separately. The full profile uses
  100 replications for sample-size experiments and 30 for sweeps. `N_REPS`
  overrides this count. Synthetic experiments use dense KRR.
- **RealCause:** `N_SAMPLES` counts dataset files, not rows. For example,
  `N_SAMPLES = 30`, `START_SAMPLE = 0` runs files 0–29 out of 100 available files.
  The full profile uses every row in each file; the quick profile reduces the
  rows and ridge grid. All RealCause KRR fits use Nyström approximation with
  up to 700 landmarks and landmark seed 700.

RealCause reports sampled-effect MSE and, for at least three full-profile files,
noise-adjusted error. The same estimated noise variance is subtracted from every
method, with joint jackknife standard errors. See the
[data and correction notes](data/realcause/SOURCE.md).

The bundled data are the LaLonde CPS samples from
[RealCause](https://github.com/bradyneal/realcause) by
[Neal, Huang, and Raghupathi (2021)](https://arxiv.org/abs/2011.15007).
No separate data download or generator training is needed.

## Repository Layout

| Path | Contents |
| --- | --- |
| `UCATE_*_Demo.ipynb` | The two experiment notebooks |
| `ours/imputation.py`, `plugin.py`, `dr.py`, `rlearner.py` | Ours and baseline estimators |
| `ours/krr.py`, `kernels.py`, `model_selection.py` | Shared solvers, kernels, and cross-validation |
| `ours/synthetic.py`, `realcause.py` | Data generation and preparation |
| `ours/*_experiments.py`, `*_reporting.py` | Experiment runners and result summaries |
| `data/realcause/` | RealCause samples, source information, and upstream license |
| `paper_results/` | Saved scores, summaries, and run settings |
| `outputs/` | Local runs and checkpoints; ignored by Git |

## Command Line

Run the 50-file RealCause experiment with four workers:

```bash
python scripts/run_realcause.py --profile full --n-samples 50 --workers 4
```

Run checks or build a submission archive:

```bash
python -m unittest discover -s tests -v
python scripts/build_submission.py
```

The archive is written to `outputs/UCATE_Camera_Ready.zip` and contains the
notebooks, source, data, and paper results, excluding local run history.
