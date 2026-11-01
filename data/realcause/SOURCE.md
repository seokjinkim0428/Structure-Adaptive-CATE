# RealCause Lalonde CPS

`lalonde_cps.zip` contains the original sample CSVs numbered 0–99. The loader
reads them directly from the archive; extraction and generator training are
unnecessary. Each file has 16,177 rows and eight covariates. Outcome/effect
columns, including `ite`, are excluded from model inputs.

Source: [RealCause](https://github.com/bradyneal/realcause), associated with
[Neal, Huang and Raghupathi (2021)](https://arxiv.org/abs/2011.15007).
The upstream MIT license is included as `LICENSE`. Dataset bytes are unchanged;
run manifests record each scored CSV's SHA-256 hash.

Noise correction uses the across-realization variance of `y1-y0` at each fixed
test covariate, averaged over test rows and divided by each realization's squared
training-outcome range. It is a conditional variance estimate, not the marginal
variance across covariates. The same correction is applied to all methods;
paired differences are unchanged. Joint jackknife errors account for reuse of
the selected files. Negative adjusted estimates are retained. Inference
conditions on the fitted generator, covariates and split.
