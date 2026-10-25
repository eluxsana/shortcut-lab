# Methods and things this experiment cannot tell you

## Data-generating process

The binary label is sampled uniformly from −1 and +1. The stable feature is `y + Normal(0, 1)`. The shortcut is `s*y + Normal(0, 0.15)`, where `s` is +1 with the selected alignment probability and −1 otherwise. Labels are converted to 0/1 for logistic loss. No feature is a sample identifier.

The training seed is 17 and the test seed is 9281. Each training dataset has 3,000 examples. Each test distribution has 4,000 examples. The alignment sweep is 0%, 5%, …, 100%.

For the same seed, changing alignment preserves the labels, stable feature, uniform alignment draws, and shortcut-noise draws. This common-random-numbers design makes changes easier to compare. The 21 test sets overlap by construction; their scores are not 21 independent replications. The realized alignment fraction can differ slightly from the requested probability.

## Training choices

1. **Shortcut-rich:** train with 95% alignment and both features.
2. **Balanced:** train with 50% alignment and both features. This is a controlled data intervention, not Group DRO, IRM, or an automatic robustness technique.
3. **Stable-only:** train with 95% alignment, setting the shortcut feature to zero. This uses privileged knowledge of the stable feature.

Every model is logistic regression with a bias term. Training uses 240 full-batch gradient steps with learning rate 0.25 and L2 coefficient 0.01 on the two feature weights, excluding the bias. Weights start at zero. The prediction threshold is logit ≥ 0. No hyperparameter search was performed on the test sweep. Convergence to the exact optimum is not claimed; loss traces are included.

The analytic stable-only Bayes accuracy for this constructed feature is Φ(1), about 84.13%. This is a useful sanity reference, not an expected exact finite-sample score. The trained stable-only model scores 83.2% on this test seed.

## Measurement

Accuracy is reported overall, on shortcut-aligned examples, and on conflicting examples. At the endpoints, one group is empty; its accuracy and interval are null, displayed as an em dash. Counts remain visible in the raw JSON.

The overall error bars are pointwise 95% Wilson score intervals for a binomial proportion conditional on the fitted model and the specified data-generating distribution. They do not include uncertainty from the training seed or training procedure, do not measure a paired difference between models, and are not simultaneous confidence bands for the curve. The slider does not interpolate results or retrain a model.

## Limits

- One training seed and one held-out latent sample. Multi-seed means and intervals are future work, not something established here.
- Two synthetic features, binary labels, and logistic regression. No evidence about LLM or real-world deployment robustness.
- The intervention is chosen with full knowledge of the generator. Real datasets rarely identify the causal feature this conveniently.
- The test sweep is a controlled illustration, not an untouched competitive benchmark.
- Floating-point rounding may differ slightly across Python/platform versions. The full configuration and source hash are recorded; the hash checks integrity, not authorship or provenance by itself.

## Reproduce and verify

`python3 train.py --out my-report` generates both the raw JSON and the offline report. `python3 -m unittest -v` checks numerical gradients, stable-only feature exclusion, deterministic split construction, common random numbers, a reference Wilson interval, empty groups, training-loss improvement, shift behavior, and extreme-logit stability.

The website has no third-party script, font, telemetry, or model dependency. Viewing it does not send evaluation examples to a service. Standard GitHub Pages access logging, if hosted there, is outside this repository's control.
