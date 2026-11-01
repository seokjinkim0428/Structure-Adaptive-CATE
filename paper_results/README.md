# Paper results

| Experiment | Summary | Individual scores |
| --- | --- | --- |
| Synthetic | [Mean MISE and SE](synthetic/summary.csv) | [Replications](synthetic/replications.csv) |
| RealCause | [MSE and SE](realcause/summary.csv) | [Dataset files](realcause/samples.csv) |
| RealCause, noise-adjusted | [Adjusted error and SE](realcause/realcause_noise_summary.csv) | [Joint jackknife](realcause/realcause_noise_jackknife.csv) |

Synthetic results cover 60 conditions and 10,650 method/replication scores.
RealCause results use files 0–49 and Nyström KRR with up to 700 landmarks and
landmark seed 700. The noise correction uses these same 50 files; see
[the correction notes](../data/realcause/SOURCE.md).

To reproduce the experiment settings:

- [Synthetic notebook](../UCATE_Synthetic_Demo.ipynb): `PROFILE = "full"`,
  `SETTING = "all"`, `STUDY = "all"`, `N_REPS = None`.
- [RealCause notebook](../UCATE_RealCause_Demo.ipynb): `PROFILE = "full"`,
  `N_SAMPLES = 50`, `START_SAMPLE = 0`.

Each experiment folder includes a `manifest.json` with settings, seeds,
dependency versions, and source hashes from the saved run. New runs are written
under `outputs/`, leaving these reference results in place.
